#!/usr/bin/env python3
"""Rewrite each spec's ``python`` block so it mirrors the ctypes signature.

``_aclnn_ctypes.py`` is the reference: parameter order, which parameters are
positional, which carry defaults, and the default values themselves are the
published API.  Every thin backend derives its Python wrapper from the spec's
``python`` block, so any drift there silently changes the call shape of an
operator that the caller did not ask to change.

This tool recomputes, per spec:

* ``python.positional`` -- the ctypes positional parameters the spec carries;
* ``python.defaults``   -- the ctypes default text for every parameter that has
  one (positional *or* keyword-only);
* ``python.required``   -- keyword-only parameters ctypes requires.

It never invents a value: parameters that ctypes does not have are reported as
manual work instead of being guessed, and spec-local keys (``pre``,
``ignored``, ``hidden``, ``return_code``, ``derive_chunk_indices``) are left
untouched.  ``op_api_parity.py`` is the check that then proves the result.

Usage::

    python tools/sync_spec_python.py --check    # report drift, exit 1 if any
    python tools/sync_spec_python.py --write    # rewrite the specs
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent.parent
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
SPEC_DIR = SETUP_DIR / "op_specs"
REFERENCE = OPS_DIR / "_aclnn_ctypes.py"


def reference_signatures() -> dict[str, dict]:
    """name -> {positional, keyword_only, defaults} for the ctypes module."""

    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("npu_")):
            continue
        args: ast.arguments = node.args
        positional = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
        defaults: dict[str, str] = {}
        for name, value in zip(positional[-len(args.defaults):], args.defaults):
            defaults[name] = _default_text(value)
        for name, value in zip(args.kwonlyargs, args.kw_defaults):
            if value is not None:
                defaults[name.arg] = _default_text(value)
        out[node.name] = {
            "positional": positional,
            "keyword_only": [a.arg for a in args.kwonlyargs],
            "defaults": defaults,
        }
    return out


def _default_text(node: ast.expr) -> str:
    """Render a default the way the specs spell it (double-quoted strings)."""

    try:
        value = ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)
    if isinstance(value, str):
        return json.dumps(value)
    return ast.unparse(node)


def plan(spec: dict, reference: dict) -> tuple[dict, list[str]]:
    """Return (new python block, manual-work notes) for one spec."""

    py = dict(spec.get("python", {}))
    arg_names = [a["name"] for a in spec["args"] if a["kind"] != "out_tensor"]
    hidden = set(py.get("hidden", ()))
    ignored = set(py.get("ignored", ()))
    ref_positional = [p for p in reference["positional"]
                      if p in arg_names and p not in hidden]
    notes: list[str] = []

    missing = [p for p in reference["positional"] + reference["keyword_only"]
               if p not in arg_names and p not in ignored]
    if missing:
        notes.append(
            f"ctypes parameters absent from the spec: {missing} "
            f"(add them to `args`, usually as cpp_only)")
    extra = [name for name in arg_names
             if name not in reference["positional"]
             and name not in reference["keyword_only"]
             and name not in hidden]
    if extra:
        notes.append(
            f"spec parameters absent from ctypes: {extra} "
            f"(hide them, or the wrapper accepts arguments the reference does not)")

    # `ignored` parameters are accepted (and dropped) by the wrapper without
    # reaching the aclnn call, so they are absent from `args` but still need
    # ctypes' default -- otherwise the wrapper would default them to None and
    # change a `False` default into "unset".
    defaults = {name: text for name, text in reference["defaults"].items()
                if (name in arg_names or name in ignored) and name not in hidden}
    required = [name for name in reference["keyword_only"]
                if name in arg_names and name not in defaults and name not in hidden]

    py["positional"] = ref_positional
    if defaults:
        py["defaults"] = defaults
    else:
        py.pop("defaults", None)
    if required:
        py["required"] = required
    else:
        py.pop("required", None)
    return py, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not (args.write or args.check):
        parser.error("choose --check or --write")

    reference = reference_signatures()
    changed = 0
    problems = 0
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        name = spec["python_name"]
        if spec.get("internal"):
            # Internal launcher ops (the shared conv1d ABI) have no public
            # Python surface to keep in step with ctypes.
            continue
        if name not in reference:
            print(f"?? {name}: not in {REFERENCE.name}")
            problems += 1
            continue
        new_python, notes = plan(spec, reference[name])
        if notes:
            problems += 1
            print(f"!! {name}")
            for note in notes:
                print(f"     {note}")
        if new_python != spec.get("python", {}):
            changed += 1
            if args.check:
                print(f"-- {name}")
                for key in ("positional", "defaults", "required"):
                    old, new = spec.get("python", {}).get(key), new_python.get(key)
                    if old != new:
                        print(f"     {key}: {old} -> {new}")
            else:
                spec["python"] = new_python
                path.write_text(
                    json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
                print(f"~~ {name}: python block synced")

    print()
    print(f"specs with drift: {changed}; specs needing manual work: {problems}")
    if args.check:
        return 1 if (changed or problems) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
