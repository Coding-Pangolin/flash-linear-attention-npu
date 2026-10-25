"""Prepare explicit public-input ATK cases from the frozen user model manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
PUBLIC_INPUTS = (
    "q k v g beta scale layout chunk_size epsilon use_qk_l2norm_in_kernel "
    "use_gate_in_kernel use_beta_sigmoid_in_kernel allow_neg_eigval safe_gate "
    "lower_bound use_exp2 a_log dt_bias cu_seqlens chunk_indices backward_mode"
).split()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def public_inputs(case):
    values = [item[0] if isinstance(item, list) else item for item in case["inputs"]]
    if [item["name"] for item in values] != PUBLIC_INPUTS:
        raise ValueError("Unexpected public input order")
    return {item["name"]: item for item in values}


def model_cases(manifest, include_h2=False):
    cases = copy.deepcopy(manifest["cases"])
    if [c["id"] for c in cases] != list(range(1, 33)):
        raise ValueError("Expected source case ids 1..32")
    if include_h2:
        for index, group in enumerate(manifest["optional_h2"]["groups"], 33):
            cu = manifest["groups"][group]
            cases.append(dict(id=index, B=1, H=2, T=cu[-1], D=128,
                              group=group, sequences=len(cu)-1))
    for case in cases:
        cu = manifest["groups"][case["group"]]
        if (case["B"] != 1 or case["D"] != 128 or case["H"] not in (2, 4, 8)
                or cu[0] != 0 or cu[-1] != case["T"]
                or len(cu) != case["sequences"] + 1
                or any(type(v) is not int for v in cu)
                or any(b <= a for a, b in zip(cu, cu[1:]))):
            raise ValueError(f"Invalid source boundaries: case{case['id']}")
    return cases


def make_case(template, model, cu, profile, common, seed, ordinal):
    case = copy.deepcopy(template)
    case.update(id=10000+ordinal, default_seed=seed,
                name=f"kda_model_c{model['id']:02d}_{profile['name']}_s{seed}")
    # No performance target was supplied for these model shapes.
    case["standard"]["perf"] = "not_key"
    inputs = public_inputs(case)
    for name in ("q", "k", "v", "g"):
        inputs[name]["shape"] = [model["B"], model["H"], model["T"], model["D"]]
    inputs["g"]["dtype"] = profile["gate_dtype"]
    inputs["beta"].update(shape=[model["B"], model["H"], model["T"]],
                          dtype=common["beta_dtype"])
    attrs = dict(scale=common["scale"], layout="BNSD", chunk_size=64,
                 epsilon=common["epsilon"], use_qk_l2norm_in_kernel=True,
                 use_gate_in_kernel=True, safe_gate=profile["safe_gate"],
                 lower_bound=profile["lower_bound"])
    for name in ("use_beta_sigmoid_in_kernel", "allow_neg_eigval", "use_exp2", "backward_mode"):
        attrs[name] = common[name]
    for name, value in attrs.items():
        inputs[name]["range_values"] = value
    inputs["a_log"]["shape"] = [model["H"]]
    inputs["dt_bias"]["shape"] = [model["H"] * model["D"]]
    # ATK encodes attrs as one InputCaseConfig per element, not a list in
    # range_values. Preserve the executor's direct host integer-list input.
    cu_items = []
    for value in cu:
        item = copy.deepcopy(inputs["cu_seqlens"])
        item.update(dtype="int", range_values=value)
        cu_items.append(item)
    case["inputs"][PUBLIC_INPUTS.index("cu_seqlens")] = cu_items
    inputs["chunk_indices"]["range_values"] = "null"
    return case


def prepare(output, config, include_h2=False, selected=None, selected_profiles=None):
    manifest_path = OP_DIR / "model_cases/manifest.json"
    template_path = OP_DIR / "atk_chunk_kda_fwd_prepare.json"
    manifest, settings = load(manifest_path), load(config)
    cases = model_cases(manifest, include_h2)
    if selected:
        if not set(selected) <= {c["id"] for c in cases}:
            raise ValueError("Unknown model case id (H2 cases require --include-h2)")
        cases = [c for c in cases if c["id"] in selected]
    profiles = settings["profiles"]
    if selected_profiles:
        if not set(selected_profiles) <= {p["name"] for p in profiles}:
            raise ValueError("Unknown profile")
        profiles = [p for p in profiles if p["name"] in selected_profiles]
    seeds = settings["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("At least one distinct integer seed is required")
    if settings["common"]["backward_mode"] != "save":
        raise ValueError("Model accuracy suite requires all 13 outputs: backward_mode=save")
    templates = {c["id"]: c for c in load(template_path)}
    names = [p["name"] for p in profiles]
    if len(set(names)) != len(names) or not profiles:
        raise ValueError("Empty or duplicate profile names")
    for profile in profiles:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", profile["name"]):
            raise ValueError("Profile names must be safe filenames")
        if profile["gate_dtype"] not in ("bf16", "fp32"):
            raise ValueError("Unsupported gate dtype")
        if type(profile["safe_gate"]) is not bool:
            raise ValueError("safe_gate must be a JSON boolean")
        if profile["safe_gate"] and not -5 <= profile["lower_bound"] < 0:
            raise ValueError("safe_gate requires lower_bound in [-5, 0)")
    if settings["common"]["beta_dtype"] not in ("bf16", "fp32"):
        raise ValueError("Unsupported beta dtype")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping, suites = [], []
    for profile_index, profile in enumerate(profiles):
        accuracy, performance = [], []
        template = templates[profile["template_id"]]
        for model in cases:
            cu = manifest["groups"][model["group"]]
            chunks = sum((b-a+63)//64 for a, b in zip(cu, cu[1:]))
            for seed in seeds:
                ordinal = profile_index * 1000 + (model["id"] - 1) * len(seeds) + seeds.index(seed)
                entry = make_case(template, model, cu, profile, settings["common"], seed, ordinal)
                accuracy.append(entry)
                if seed == seeds[0]:
                    perf = copy.deepcopy(entry)
                    perf["name"] += "_perf"
                    performance.append(perf)
                mapping.append(dict(case_id=model["id"], group=model["group"], H=model["H"],
                                    T=model["T"], sequences=model["sequences"], chunks=chunks,
                                    profile=profile["name"], seed=seed, atk_id=entry["id"],
                                    atk_name=entry["name"], expected_route="upstream_fallback"))
        files = {}
        for kind, records in (("accuracy", accuracy), ("performance", performance)):
            path = output / f"atk_chunk_kda_fwd_prepare_model_{profile['name']}_{kind}.json"
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            files[kind] = dict(file=path.name, count=len(records), sha256=sha256(path))
        suites.append(dict(profile=profile, files=files))
    with (output / "case_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    plan = dict(schema_version=1, platforms=["A2", "A5"], model_cases=len(cases),
                accuracy_cases=len(mapping), performance_cases=len(cases)*len(profiles),
                seeds=seeds, model_parameters_confirmed=settings["model_parameters_confirmed"],
                calibration_status=settings["calibration_status"],
                source_sha256=manifest["source_sha256"], manifest_sha256=sha256(manifest_path),
                config_sha256=sha256(config), settings=settings, templates_sha256=sha256(template_path),
                executor_sha256=sha256(OP_DIR / "executor_chunk_kda_fwd_prepare.py"),
                reference="unchanged executor FP64 CPU; mixed_tolerance_bm; all 13 outputs",
                suites=suites)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=OP_DIR / "model_cases/profiles.json")
    parser.add_argument("--include-h2", action="store_true")
    parser.add_argument("--cases", help="Comma-separated model ids; omitted means full set")
    parser.add_argument("--profiles", help="Comma-separated profile names; omitted means all")
    args = parser.parse_args()
    plan = prepare(args.out, args.config, args.include_h2,
                   [int(v) for v in args.cases.split(",")] if args.cases else None,
                   args.profiles.split(",") if args.profiles else None)
    print(json.dumps({k:plan[k] for k in ("model_cases", "accuracy_cases", "performance_cases",
                                        "model_parameters_confirmed", "calibration_status")}, indent=2))


if __name__ == "__main__":
    main()
