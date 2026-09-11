#!/usr/bin/env python3
"""Check each spec's aclnn argument list against the ctypes implementation.

``op_api_parity.py`` compares the *Python* surface of the backends.  This one
compares the layer below it: the arguments a spec hands to the aclnn entry
point must be the arguments the ctypes reference hands to the same entry point,
in the same order and of the same kind.

That gap is not hypothetical.  Upstream #390 rewrote ``aclnnCausalConv1d``
(four ``aclIntArray`` metadata slots became ``aclTensor``, ``activationMode``
became a ``const char*``, ``nullBlockId``/``maxQueryLen`` appeared) and the
spec still described the old list; nothing in the tree noticed until a kernel
call died.  ``op_abi_validate.py`` can catch this class of change, but it needs
the aclnn *header*, which only exists next to a matching OPP.  This tool reads
the implementation instead, so it works anywhere the tree does.

How the kinds are read: the aclnn call is built either from a ``lambda ctx:
[...]`` or from a local ``build_args`` function; the arguments are
``ctx.tensor(...)`` / ``ctx.int_array(...)`` / ``ctypes.c_int64(...)`` and
friends.  When the implementation delegates to a module-level helper that owns
the ``_call_aclnn`` (the conv1d family does), the helper's parameter order plus
the caller's keyword arguments are used.  Anything that cannot be resolved
statically is reported as UNRESOLVED rather than guessed.

Usage::

    python tools/op_abi_parity.py            # exit 1 on any mismatch
    python tools/op_abi_parity.py --json
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
SPEC_DIR = SETUP_DIR / "op_specs"

# aclnn prototype kind -> the spec's vocabulary
CTYPES_KINDS = {
    "c_int64": "int64",
    "c_double": "double",
    "c_float": "float",
    "c_bool": "bool",
    "cast": "char_ptr",  # ctypes.cast(buffer, ctypes.c_char_p)
    "c_char_p": "char_ptr",  # ctypes.c_char_p(text.encode())
    "c_void_p": "tensor",  # a null aclTensor*: the reference spells it this way
}
SPEC_ALIASES = {"optional_tensor": "tensor", "out_tensor": "tensor",
                "cpu_int_array": "int_array"}


def _call_kind(node: ast.expr) -> str | None:
    """Map one argument expression in the aclnn list to a kind."""

    if isinstance(node, ast.Name):
        # A bare local such as an activation/layout buffer built earlier.
        return None
    if isinstance(node, ast.IfExp):
        # `ctx.tensor(out, 'out') if out is not None else ctypes.c_void_p(0)`:
        # both branches are the same aclnn slot (a null tensor when absent).
        return _call_kind(node.body) or _call_kind(node.orelse)
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "ctx":
            if func.attr in {"tensor", "int_array"}:
                return func.attr
        if isinstance(func.value, ast.Name) and func.value.id == "ctypes":
            return CTYPES_KINDS.get(func.attr)
    # A helper such as nd_tensor(ctx, x, 'x'): treat it as a tensor view.
    if isinstance(func, ast.Name) and "tensor" in func.id:
        return "tensor"
    return None


def _local_kinds(node: ast.AST) -> dict[str, str]:
    """Kinds of locals that were assigned from a mapped call in *node*."""

    kinds: dict[str, str] = {}
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign):
            continue
        kind = _call_kind(stmt.value)
        if kind is None:
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                kinds[target.id] = kind
    return kinds


def _literal_lengths(node: ast.AST) -> dict[str, int]:
    """Lengths of locals bound to a literal tuple/list (name pools)."""

    lengths: dict[str, int] = {}
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and isinstance(
                    stmt.value, (ast.Tuple, ast.List)):
                lengths[target.id] = len(stmt.value.elts)
    return lengths


def _expand(argument: ast.expr, kind: str | None,
            locals_: dict[str, str],
            literals: dict[str, int] | None = None) -> list[str | None]:
    """One list element -> one or more kinds.

    ``*[helper(ctx, out) for out in (...)]`` is a fixed-length expansion: the
    names tuple right there says how many tensors it contributes.
    """

    if isinstance(argument, ast.Starred):
        value = argument.value
        # `*[helper(ctx, x) for x in ...]` / `*（helper(...) for ...)`
        if isinstance(value, (ast.ListComp, ast.GeneratorExp)):
            count = None
            for generator in value.generators:
                if not (isinstance(generator.iter, ast.Call)
                        and getattr(generator.iter.func, "id", "") == "zip"
                        and len(generator.iter.args) == 2):
                    continue
                names = generator.iter.args[1]
                if isinstance(names, ast.Tuple):
                    count = len(names.elts)
                elif isinstance(names, ast.Name) and literals:
                    count = literals.get(names.id)
            element_kind = _call_kind(value.elt)
            if count is not None and element_kind is not None:
                return [element_kind] * count
        return [None]
    if isinstance(argument, ast.Name) and kind is None:
        return [locals_.get(argument.id)]
    return [kind]


def _argument_list(node: ast.AST) -> list[ast.expr] | None:
    """The list of aclnn arguments produced by *node* (lambda or return)."""

    if isinstance(node, ast.Lambda):
        body = node.body
        return body.elts if isinstance(body, ast.List) else None
    if isinstance(node, ast.FunctionDef):
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
                return stmt.value.elts
    return None


def _aclnn_call(node: ast.AST) -> tuple[str, ast.AST] | None:
    """(aclnn entry point, argument-producing node) inside *node*, if any."""

    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Call):
            continue
        func = stmt.func
        if not (isinstance(func, ast.Name) and func.id == "_call_aclnn"):
            continue
        if len(stmt.args) < 2:
            return None
        name = stmt.args[0]
        if not isinstance(name, ast.Constant):
            return None
        source = stmt.args[1]
        if isinstance(source, ast.Name):
            # build_args(ctx) defined in the same function
            for inner in ast.walk(node):
                if isinstance(inner, ast.FunctionDef) and inner.name == source.id:
                    return name.value, inner
            return name.value, ast.Constant(value=None)
        return name.value, source
    return None


def implementation_kinds() -> dict[str, dict]:
    """name -> {aclnn, kinds, unresolved} from the ctypes module."""

    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    helpers: dict[str, list[str]] = {}
    helper_entry: dict[str, str] = {}
    for name, node in functions.items():
        found = _aclnn_call(node)
        if found is None:
            continue
        _, produced = found
        arguments = _argument_list(produced)
        if arguments is None:
            continue
        locals_ = _local_kinds(node)
        literals = _literal_lengths(node)
        kinds: list[str | None] = []
        for argument in arguments:
            kinds.extend(_expand(argument, _call_kind(argument), locals_,
                                 literals))
        helpers[name] = kinds
        helper_entry[name] = found[0]

    result: dict[str, dict] = {}
    for name, node in functions.items():
        found = _aclnn_call(node)
        if found is None:
            # Delegating to a helper that owns the _call_aclnn (conv1d family).
            delegations = [
                call for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in helpers
            ]
            if not delegations:
                result[name] = {"aclnn": None, "kinds": None,
                                "note": "no aclnn call found"}
                continue
            call = delegations[0]
            helper_name = call.func.id
            # The helper's own aclnn list *is* the ABI, in the helper's parameter
            # order, which is the order its callers fill it in.  Comparing
            # against that beats re-deriving the mapping from keyword arguments.
            result[name] = {"aclnn": helper_entry[helper_name],
                            "kinds": helpers[helper_name],
                            "note": f"via {helper_name}"}
            continue
        aclnn, produced = found
        arguments = _argument_list(produced)
        locals_ = _local_kinds(node)
        literals = _literal_lengths(node)
        kinds_list: list[str | None] | None = None
        if arguments is not None:
            kinds_list = []
            for argument in arguments:
                kinds_list.extend(_expand(argument, _call_kind(argument),
                                          locals_, literals))
        result[name] = {
            "aclnn": aclnn,
            "kinds": kinds_list,
        }
    return result


def spec_kinds() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        kinds = [SPEC_ALIASES.get(argument["kind"], argument["kind"])
                 for argument in spec["args"]
                 if not argument.get("cpp_only")]
        out[spec["python_name"]] = {
            "aclnn": spec["aclnn_name"],
            "kinds": kinds,
            "spec": path.name,
            # An internal spec describes a launcher op; the implementation that
            # owns the aclnn call is named explicitly (conv1d's shared launcher)
            # because there is no ctypes function with the internal op's name.
            "impl": spec.get("impl", spec["python_name"]),
        }
    return out


def evaluate() -> dict:
    implementation = implementation_kinds()
    specs = spec_kinds()
    rows: list[dict] = []
    for name in sorted(specs):
        spec = specs[name]
        impl = implementation.get(spec["impl"])
        problems: list[str] = []
        if impl is None:
            problems.append(
                f"implementation {spec['impl']!r} not found in the ctypes module")
        elif impl.get("kinds") is None:
            problems.append(f"aclnn argument list unresolved ({impl.get('note')})")
        else:
            expected = spec["kinds"]
            observed = impl["kinds"]
            if None in observed:
                problems.append("aclnn argument list unresolved")
            elif expected != observed:
                problems.append(
                    f"aclnn arguments differ:\n        spec: {expected}\n"
                    f"        impl: {observed}")
            if impl.get("aclnn") not in (None, spec["aclnn"]):
                problems.append(
                    f"aclnn entry point differs: spec {spec['aclnn']!r} vs "
                    f"implementation {impl['aclnn']!r}")
        rows.append({"op": name, "spec": spec["spec"], "problems": problems})
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        problems = [row for row in report["rows"] if row["problems"]]
        print(f"specs checked: {len(report['rows'])}")
        print(f"mismatched   : {len(problems)}")
        print()
        for row in problems:
            print(f"{row['op']}  [{row['spec']}]")
            for problem in row["problems"]:
                print(f"    - {problem}")
        if not problems:
            print("ABI MATCH: every spec describes the arguments its "
                  "implementation passes to aclnn")
    return 1 if any(row["problems"] for row in report["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
