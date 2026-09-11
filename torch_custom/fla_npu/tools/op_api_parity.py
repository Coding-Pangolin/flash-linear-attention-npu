#!/usr/bin/env python3
"""Compare the public Python signature of every backend against ctypes.

The ctypes module is the reference users have been calling: its parameter
order, which parameters carry defaults, and the default values themselves are
part of the published API.  A thin backend that changes any of those turns a
working call into a ``TypeError`` -- and because the thin path is selected
transparently by ``_get_direct_op``, the caller never asked for that change.

This tool parses the three modules with :mod:`ast` (no import, so no torch and
no NPU are needed) and reports, per operator:

* parameters missing from the backend, or present only in the backend;
* a different positional order;
* a positional parameter that is keyword-only (or vice versa);
* a default that changed, appeared, or disappeared.

Usage::

    python tools/op_api_parity.py            # compare everything, exit 1 on drift
    python tools/op_api_parity.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent.parent
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"

REFERENCE = OPS_DIR / "_aclnn_ctypes.py"
BACKENDS = {
    "stable-generated": OPS_DIR / "_stable_generated.py",
    "stable-handwritten": OPS_DIR / "_stable.py",
    # Still selectable with FLA_NPU_THIN_ABI=pybind, so its Python surface is
    # part of the published API too -- a caller who switches backends must not
    # discover that a keyword was renamed.
    "pybind": OPS_DIR / "_thin.py",
}

def _defaults(node: ast.arguments) -> dict[str, str]:
    """Map every parameter with a default to its unparsed default text."""

    positional = list(node.posonlyargs) + list(node.args)
    out: dict[str, str] = {}
    for name, value in zip(positional[-len(node.defaults):], node.defaults):
        out[name.arg] = ast.unparse(value)
    for name, value in zip(node.kwonlyargs, node.kw_defaults):
        if value is not None:
            out[name.arg] = ast.unparse(value)
    return out


def _signature(node: ast.FunctionDef) -> dict:
    node_args: ast.arguments = node.args
    positional = [a.arg for a in node_args.posonlyargs] + \
        [a.arg for a in node_args.args]
    kwonly = [a.arg for a in node_args.kwonlyargs]
    return {
        "positional": positional,
        "keyword_only": kwonly,
        "defaults": _defaults(node_args),
    }


def signatures(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("npu_"):
            out[node.name] = _signature(node)
    return out


def _compare(name: str, reference: dict, backend: dict) -> list[str]:
    problems: list[str] = []
    ref_pos = reference["positional"]
    got_pos = backend["positional"]
    if ref_pos != got_pos:
        missing = [p for p in ref_pos if p not in got_pos]
        extra = [p for p in got_pos if p not in ref_pos]
        if missing:
            problems.append(f"positional parameters missing: {missing}")
        if extra:
            problems.append(f"unexpected positional parameters: {extra}")
        if not missing and not extra:
            problems.append(
                f"positional order differs: ctypes {ref_pos} vs backend {got_pos}")

    ref_kw = reference["keyword_only"]
    got_kw = backend["keyword_only"]
    drifted_kind = [p for p in ref_pos if p in got_kw]
    if drifted_kind:
        problems.append(
            f"ctypes positional became keyword-only: {drifted_kind} "
            f"(a positional call now raises TypeError)")
    became_positional = [p for p in ref_kw if p in got_pos]
    if became_positional:
        problems.append(
            f"ctypes keyword-only became positional: {became_positional}")
    missing_kw = [p for p in ref_kw if p not in got_kw and p not in got_pos]
    if missing_kw:
        problems.append(f"keyword-only parameters missing: {missing_kw}")
    extra_kw = [p for p in got_kw if p not in ref_kw and p not in ref_pos]
    if extra_kw:
        problems.append(f"unexpected keyword-only parameters: {extra_kw}")

    for parameter, default in reference["defaults"].items():
        if parameter not in backend["defaults"]:
            problems.append(
                f"default lost on {parameter!r} (ctypes {default})")
        elif backend["defaults"][parameter] != default:
            problems.append(
                f"default changed on {parameter!r}: "
                f"ctypes {default} vs backend {backend['defaults'][parameter]}")
    for parameter, default in backend["defaults"].items():
        if parameter not in reference["defaults"]:
            problems.append(
                f"default added on {parameter!r} (backend {default})")
    return problems


def evaluate() -> dict:
    reference = signatures(REFERENCE)
    backends = {label: signatures(path) for label, path in BACKENDS.items()}
    rows: list[dict] = []
    for name in sorted(reference):
        found = False
        for label, table in backends.items():
            if name not in table:
                continue
            found = True
            # Every backend that exposes the operator is compared, not just the
            # one that happens to answer first: FLA_NPU_THIN_ABI switches
            # between them, so a caller must not hit a renamed keyword by
            # changing a flag.
            rows.append({
                "op": name,
                "backend": label,
                "problems": _compare(name, reference[name], table[name]),
            })
        if not found:
            rows.append({"op": name, "backend": None,
                         "problems": ["no stable backend signature found"]})
    return {"rows": rows, "reference": str(REFERENCE.name)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate()
    rows = report["rows"]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        drifted = [row for row in rows if row["problems"]]
        print(f"operators compared: {len(rows)}")
        print(f"drifted          : {len(drifted)}")
        print()
        for row in rows:
            if not row["problems"]:
                continue
            print(f"{row['op']}  [{row['backend']}]")
            for problem in row["problems"]:
                print(f"    - {problem}")
        if not drifted:
            print("SIGNATURES MATCH: every backend exposes the ctypes call shape")
    return 1 if any(row["problems"] for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
