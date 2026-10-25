"""Portable ATK delivery manifest helpers; no tensors or reference cache."""
from __future__ import annotations

import copy
import itertools
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OP = "chunk_kda_fwd_prepare"
SEEDS = (3, 17, 41)
# Canonical id/name/original-seed/public-input records at frozen826c983.
ORIGINAL_CONTRACT_SHA256 = "05de73b8796b0a7c040f487bc138cee2d7c4cb425e70a3a17092c57b688e58a4"
STANDARD = {"acc": {"cv_fused_double_benchmark": {
    "max_re_ratio": 5, "avg_re_ratio": 1.5,
    "root_mean_squared_ratio": 1.5}}, "perf": "not_key"}
PUBLIC_INPUTS = ("q k v g beta scale layout chunk_size epsilon "
    "use_qk_l2norm_in_kernel use_gate_in_kernel use_beta_sigmoid_in_kernel "
    "allow_neg_eigval safe_gate lower_bound use_exp2 a_log dt_bias "
    "cu_seqlens chunk_indices backward_mode").split()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inputs(case):
    items = [x[0] if isinstance(x, list) else x for x in case["inputs"]]
    if [x["name"] for x in items] != PUBLIC_INPUTS:
        raise ValueError("Expected exactly 21 public inputs in interface order")
    for container, item in zip(case["inputs"], items):
        if item["type"] == "attrs" and not isinstance(container, list):
            raise ValueError(f"ATK attrs input {item['name']} requires a list, including null")
    return {x["name"]: x for x in items}


def attr(case, name):
    return inputs(case)[name]["range_values"]


def signature(case, target=False):
    """Symbolic template tuple, NOT an observed numeric tiling key."""
    i = inputs(case)
    gate = 0 if not attr(case, "use_gate_in_kernel") else (
        2 if attr(case, "safe_gate") else 1)
    beta = 0 if not attr(case, "use_beta_sigmoid_in_kernel") else (
        2 if attr(case, "allow_neg_eigval") else 1)
    return (i["g"]["dtype"], i["beta"]["dtype"],
            int(attr(case, "use_qk_l2norm_in_kernel")), beta, gate,
            int(attr(case, "use_exp2")), int(target or gate == 2),
            attr(case, "backward_mode"))


def all_signatures():
    return {(g, b, n, bm, gm, e, int(gm == 2), m)
            for g, b, n, bm, gm, e, m in itertools.product(
                ("bf16", "fp32"), ("bf16", "fp32"), (0, 1),
                (0, 1, 2), (0, 1, 2), (0, 1),
                ("none", "recompute", "save", "forward"))}


def expand_seeds(cases, seeds=SEEDS):
    expanded = []
    for index, case in enumerate(cases):
        for seed in seeds:
            item = copy.deepcopy(case)
            item.update(id=index * len(seeds) + seeds.index(seed),
                        default_seed=seed, name=f"{case['name']}_seed{seed}")
            expanded.append(item)
    return expanded


def forward_cases():
    """Supplement four layouts without replacing any original Save comparison."""
    cases = load(ROOT / f"atk_{OP}.json")
    selected = []
    for layout in ("BNSD", "BSND", "NTD", "TND"):
        case = next(c for c in cases if attr(c, "layout") == layout
                    and attr(c, "backward_mode") == "save")
        item = copy.deepcopy(case)
        item.update(id=20000+len(selected), name=case["name"]+"_forward_supplement")
        inputs(item)["backward_mode"]["range_values"] = "forward"
        selected.append(item)
    return selected


def write_json(path, value):
    if isinstance(value, list) and value and all(isinstance(c, dict) and "inputs" in c for c in value):
        # Keep one public input per line. Repeating every nullable ATK field over
        # many lines obscures meaningful changes in these frozen configurations.
        records = []
        for case in value:
            fields = []
            for key, item in case.items():
                encoded = json.dumps(item, ensure_ascii=False)
                if key == "inputs":
                    encoded = "[\n" + ",\n".join("      " + json.dumps(i, ensure_ascii=False)
                        for i in item) + "\n    ]"
                fields.append("    " + json.dumps(key) + ": " + encoded)
            records.append("  {\n" + ",\n".join(fields) + "\n  }")
        text = "[\n" + ",\n".join(records) + "\n]\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def validate():
    accuracy = load(ROOT / f"atk_{OP}.json")
    mss = load(ROOT / f"atk_{OP}_mss.json")
    if len(accuracy) != 200 or len({c["name"] for c in accuracy}) != 200:
        raise ValueError("Require 200 distinct logical precision cases")
    contracts = [{k:c[k] for k in ("id", "name", "default_seed", "inputs")} for c in accuracy]
    if hashlib.sha256(json.dumps(contracts, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != ORIGINAL_CONTRACT_SHA256:
        raise ValueError("Original200 inputs, output modes, IDs, names or seeds changed")
    structures = {json.dumps(c["inputs"], sort_keys=True) for c in accuracy}
    if len(structures) != 200:
        raise ValueError("Seed/name changes cannot count as new logical cases")
    if len(mss) != 6:
        raise ValueError("Require four runtime generic and two target key cases")
    general = [c for c in mss if "target_" not in c["name"]]
    target = [c for c in mss if "target_" in c["name"]]
    if len(general) != 4 or {(inputs(c)["g"]["dtype"],inputs(c)["beta"]["dtype"])
            for c in general} != {(g,b) for g in ("bf16","fp32") for b in ("bf16","fp32")}:
        raise ValueError("Runtime generic dtype coverage has gaps or duplicates")
    if len(target) != 2 or len({signature(c, True) for c in target}) != 2:
        raise ValueError("Two arch22 target instances are required")
    if {attr(c, "backward_mode") for c in accuracy} != {
            "none", "recompute", "save"}:
        raise ValueError("Original precision modes must not be replaced")
    forward = load(ROOT / f"atk_{OP}_forward.json")
    if forward != forward_cases():
        raise ValueError("Forward supplements changed or replaced original cases")
    for filename in (f"atk_{OP}.json", f"atk_{OP}_mss.json",
                     f"atk_{OP}_perf.json", f"atk_{OP}_paths.json", f"atk_{OP}_forward.json"):
        cases = load(ROOT / filename)
        for case in cases:
            inputs(case)
            if case["standard"] != STANDARD:
                raise ValueError(f"Dual comparator changed: {filename}: {case['name']}")
    return dict(logical_accuracy=200, precision_executions=200,
                mss_template_cases=6, numeric_keys_verified=False,
                model_performance_cases=len(load(ROOT / f"atk_{OP}_perf.json")),
                supplemental_paths=len(load(ROOT / f"atk_{OP}_paths.json")))


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
