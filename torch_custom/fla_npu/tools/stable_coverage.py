#!/usr/bin/env python3
"""Offline coverage gate for the Stable-ABI backend (no NPU required).

The question this answers is *not* "does a test pass" but "is every operator and
every scenario axis accounted for".  Three things are checked, all from files in
the tree:

1. **Adapter coverage** -- every ``npu_*`` operator that the ctypes layer
   exposes has a stable adapter, either generated (``kSchema_<op>`` in
   ``csrc_stable/generated/ops_stable_generated.inc``) or hand-written (a
   function of the same name in ``fla_npu/ops/ascendc/_stable.py``).  A missing
   adapter is a FAIL unless it is listed in ``tests/stable_coverage_baseline.json``
   with a reason, so gaps are recorded rather than silently tolerated.

2. **Scenario-axis declaration** -- which input axes an operator actually has is
   derived from its spec, never invented:

   * ``enum`` tables on ``char_ptr`` arguments -> the layout / dtype axis and the
     exact legal values (a ``char_ptr`` without a table is reported as
     UNVERIFIABLE, because nothing in the tree says which strings are legal);
   * ``cu_seqlens`` / ``chunk_indices`` arguments -> the varlen axis;
   * ``return_when`` on outputs plus boolean arguments -> the flag axis.

   ``scenarios`` (when a spec carries one) is cross-checked against those
   derived axes: a scenario naming a value the spec cannot represent is a FAIL.
   The shape is ``{"<axis>": [<legal values>], ...}`` on purpose -- a flat
   mapping stays readable in a diff, unlike a list of case objects.

3. **Spec/registration consistency** -- a registered stable op whose spec is
   missing, or a spec whose ``python_name`` is not registered, is a FAIL.

Usage::

    python tools/stable_coverage.py              # human-readable matrix
    python tools/stable_coverage.py --json       # machine-readable
    python tools/stable_coverage.py --strict     # ignore the baseline (gaps FAIL)

Exit code is 0 when there is no unexplained gap, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = SETUP_DIR / "fla_npu"
OPS_DIR = PACKAGE_DIR / "ops" / "ascendc"
SPEC_DIR = SETUP_DIR / "op_specs"
TESTS_DIR = SETUP_DIR.parent.parent / "tests"
BASELINE = TESTS_DIR / "stable_coverage_baseline.json"

_DEF_RE = re.compile(r"^def\s+(npu_[a-z0-9_]+)\s*\(", re.MULTILINE)
_SCHEMA_RE = re.compile(r"kSchema_(npu_[a-z0-9_]+)\s*=")
_LAYOUT_CODES_RE = re.compile(r"_LAYOUT_CODES\s*=\s*\{([^}]*)\}")

# Axes are attached to the arguments that actually carry them.  Names are taken
# from the ctypes signatures, not invented: these are the arguments that switch
# an operator between dense and varlen (or between decode and speculative
# decode) execution.
VARLEN_ARGS = ("cu_seqlens", "chunk_indices", "actual_seq_lengths",
               "query_start_loc")
SPEC_DECODE_ARGS = ("cache_indices", "num_accepted_tokens")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ctypes_ops() -> list[str]:
    """Ground truth: the operators the ctypes layer exposes."""

    return sorted(set(_DEF_RE.findall(_read(OPS_DIR / "_aclnn_ctypes.py"))))


def generated_ops() -> list[str]:
    inc = SETUP_DIR / "csrc_stable" / "generated" / "ops_stable_generated.inc"
    if not inc.exists():
        return []
    return sorted(set(_SCHEMA_RE.findall(_read(inc))))


def hand_written_ops() -> list[str]:
    return sorted(set(_DEF_RE.findall(_read(OPS_DIR / "_stable.py"))))


def python_wrappers() -> list[str]:
    path = OPS_DIR / "_stable_generated.py"
    if not path.exists():
        return []
    return sorted(set(_DEF_RE.findall(_read(path))))


def hand_written_layouts() -> dict[str, list[str]]:
    """``_LAYOUT_CODES`` in _stable.py, per hand-written adapter."""

    text = _read(OPS_DIR / "_stable.py")
    table: dict[str, list[str]] = {}
    for body in _LAYOUT_CODES_RE.findall(text):
        names = re.findall(r'"([A-Za-z0-9_]+)"', body)
        if names:
            # One shared table today; applied to every hand-written adapter that
            # takes a `layout` argument (checked per operator below).
            table["_shared"] = sorted(set(names))
    return table


def load_specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(_read(path))
        name = spec.get("python_name")
        if not name:
            raise SystemExit(f"{path.name}: spec has no python_name")
        spec["_path"] = path.name
        specs[name] = spec
    return specs


def load_baseline() -> dict:
    if not BASELINE.exists():
        return {"known_gaps": {}, "notes": {}}
    return json.loads(_read(BASELINE))


def _axes(spec: dict) -> dict:
    """Derive the input axes an operator really has, from its spec alone."""

    axes: dict[str, list] = {}
    unverifiable: list[str] = []
    flags: list[str] = []
    arguments = set()

    for arg in spec.get("args", []):
        kind = arg.get("kind")
        name = arg.get("name", "")
        if kind != "out_tensor":
            arguments.add(name)
        if kind == "char_ptr":
            enum = arg.get("enum")
            if enum:
                axes[name] = list(enum)
            else:
                unverifiable.append(name)
        elif kind == "bool" and not name.startswith("_"):
            flags.append(name)
        if name in VARLEN_ARGS:
            axes["varlen"] = [False, True]
        if name in SPEC_DECODE_ARGS:
            axes["spec_decode"] = [False, True]

    for output in spec.get("outputs", []):
        condition = output.get("return_when")
        if condition:
            flags.append(condition)

    if flags:
        axes["flags"] = sorted(set(flags))
    return {"axes": axes, "flags": sorted(set(flags)),
            "unverifiable": unverifiable, "arguments": arguments}


def evaluate() -> dict:
    ctypes_names = ctypes_ops()
    gen = set(generated_ops())
    hand = set(hand_written_ops())
    wrappers = set(python_wrappers())
    specs = load_specs()
    shared_layouts = hand_written_layouts().get("_shared", [])

    rows: list[dict] = []
    for name in ctypes_names:
        spec = specs.get(name)
        if name in gen:
            source = "generated"
        elif name in hand:
            source = "hand-written"
        else:
            source = None
        row = {
            "op": name,
            "source": source,
            "spec": spec["_path"] if spec else None,
            "axes": {},
            "unverifiable": [],
            "problems": [],
        }
        if source is None:
            row["problems"].append("no stable adapter")
        if spec is None and source is not None:
            row["problems"].append("adapter without spec")
        if spec is not None:
            derived = _axes(spec)
            row["axes"] = derived["axes"]
            row["unverifiable"] = derived["unverifiable"]
            if source == "hand-written" and derived["unverifiable"] and shared_layouts:
                # The hand-written adapters map the layout string to an int code
                # in _stable.py, so those values are legal even though the spec
                # carries no enum table.
                for arg_name in derived["unverifiable"]:
                    if arg_name == "layout":
                        row["axes"]["layout"] = shared_layouts
                row["unverifiable"] = [
                    name_ for name_ in derived["unverifiable"]
                    if not (name_ == "layout" and shared_layouts)
                ]
            scenarios = spec.get("scenarios") or {}
            if not isinstance(scenarios, dict):
                row["problems"].append("scenarios must be a mapping of axis -> values")
            for axis, values in scenarios.items():
                declared = row["axes"].get(axis)
                if declared is None:
                    # Derived axes are the free ones (enum tables, varlen and
                    # spec-decode argument names, gating booleans).  An operator
                    # usually has more: integer knobs such as conv1d's run_mode
                    # or head_num, whose legal values only exist in the kernel
                    # and in the unit tests.  A declaration backed by a real
                    # argument is accepted and recorded; one naming something
                    # that is not an argument at all is not.
                    if axis in derived["arguments"]:
                        row["axes"][axis] = list(values)
                        continue
                    row["problems"].append(
                        f"scenario axis {axis!r} matches no argument of the spec")
                    continue
                unknown = [v for v in values if v not in declared]
                if unknown:
                    row["problems"].append(
                        f"scenario axis {axis!r} names unsupported values {unknown}")
        rows.append(row)

    orphans = sorted((gen | hand) - set(ctypes_names))
    for name in orphans:
        rows.append({
            "op": name,
            "source": "generated" if name in gen else "hand-written",
            "spec": specs.get(name, {}).get("_path"),
            "axes": {},
            "unverifiable": [],
            "problems": ["registered but not exposed by ctypes"],
        })

    registered_hand = sorted(hand)
    registered_gen = sorted(gen)
    generated_without_wrapper = sorted(gen - wrappers)
    wrapper_without_registration = sorted(wrappers - gen - hand)

    blockers = [f"{row['op']}: {'; '.join(row['problems'])}"
                for row in rows if row["problems"]]
    blockers += [f"generated but no Python wrapper: {name}"
                 for name in generated_without_wrapper]
    blockers += [f"Python wrapper without registration: {name}"
                 for name in wrapper_without_registration]

    return {
        "ctypes_ops": sorted(ctypes_names),
        "generated": registered_gen,
        "hand_written": registered_hand,
        "rows": rows,
        "orphans": orphans,
        "generated_without_wrapper": generated_without_wrapper,
        "wrapper_without_registration": wrapper_without_registration,
        "blockers": blockers,
        "fail": bool(blockers),
    }


def render(report: dict, strict: bool, baseline: dict) -> tuple[str, bool]:
    gaps = ({} if strict else baseline.get("known_gaps", {}))
    lines: list[str] = []
    unexplained: list[str] = []
    waived: list[str] = []

    lines.append(f"ctypes operators: {len(report['ctypes_ops'])}")
    lines.append(
        f"stable adapters : {len(report['generated'])} generated + "
        f"{len(report['hand_written'])} hand-written")
    lines.append("")
    lines.append(f"{'operator':<46} {'source':<13} {'axes':<6} status")
    lines.append("-" * 90)

    for row in report["rows"]:
        problems = list(row["problems"])
        if problems and row["op"] in gaps:
            waived.append(f"{row['op']}: {'; '.join(problems)} -- {gaps[row['op']]}")
            status = "KNOWN GAP"
        elif problems:
            unexplained.append(f"{row['op']}: {'; '.join(problems)}")
            status = "FAIL: " + "; ".join(problems)
        elif row["unverifiable"]:
            status = "OK (unverifiable: " + ", ".join(row["unverifiable"]) + ")"
        else:
            status = "OK"
        lines.append(
            f"{row['op']:<46} {str(row['source'] or '-'):<13} "
            f"{len(row['axes']):<6} {status}")

    lines.append("")
    if waived:
        lines.append("recorded gaps (baseline):")
        lines.extend(f"  - {item}" for item in waived)
        lines.append("")
    if unexplained:
        lines.append("UNEXPLAINED GAPS:")
        lines.extend(f"  - {item}" for item in unexplained)
        return "\n".join(lines), False

    lines.append("ALL COVERED: every ctypes operator has a stable adapter "
                 "(recorded gaps only)")
    return "\n".join(lines), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    parser.add_argument("--strict", action="store_true",
                        help="do not honour tests/stable_coverage_baseline.json")
    parser.add_argument("--axes", action="store_true",
                        help="list the derived scenario axes and their legal values")
    args = parser.parse_args()

    report = evaluate()
    baseline = load_baseline()
    gaps = ({} if args.strict else baseline.get("known_gaps", {}))
    unexplained = [b for b in report["blockers"]
                   if b.split(":", 1)[0] not in gaps]
    if args.axes:
        for row in report["rows"]:
            if not row["axes"] and not row["unverifiable"]:
                continue
            print(f"{row['op']}:")
            for axis, values in sorted(row["axes"].items()):
                print(f"  {axis:<18} {values}")
            for axis in row["unverifiable"]:
                print(f"  {axis:<18} UNVERIFIABLE (no enum table in the spec)")
        print()
    if args.json:
        report["baseline"] = baseline
        report["unexplained"] = unexplained
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if unexplained else 0

    text, ok = render(report, args.strict, baseline)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
