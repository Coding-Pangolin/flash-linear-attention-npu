#!/usr/bin/env python3
"""Apply a spec end-to-end: cpp + pybind + _thin wrapper + whitelist."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # torch_custom/fla_npu
SPEC_DIR = ROOT / "op_specs"
CSRC = ROOT / "csrc_thin" / "src"
THIN = ROOT / "fla_npu" / "ops" / "ascendc"


def cpp_type(kind: str, name: str) -> str:
    if kind == "tensor":
        return f"const at::Tensor& {name}"
    if kind == "optional_tensor":
        return f"const c10::optional<at::Tensor>& {name}"
    if kind == "int_array":
        return f"const std::vector<int64_t>& {name}"
    if kind == "char_ptr":
        return f"const std::string& {name}"
    return {"int64": "int64_t", "bool": "bool", "double": "double",
            "float": "float"}[kind] + f" {name}"


def patch_pybind(spec: dict) -> None:
    name = spec["python_name"]
    args = [a for a in spec["args"] if a["kind"] != "out_tensor"]
    path = CSRC / "pybind.cpp"
    text = path.read_text(encoding="utf-8")
    if f"at::Tensor {name}(" in text:
        return
    params = ",\n    ".join(cpp_type(a["kind"], a["name"]) for a in args)
    decl = (f"\nat::Tensor {name}(\n    {params},\n    uint64_t stream);\n"
            f"\n}}  // namespace fla_npu_thin")
    text = text.replace("}  // namespace fla_npu_thin", decl, 1)
    argnames = ",\n      ".join(
        f'py::arg("{a["name"]}")' for a in args)
    block = (f'  m.def(\n      "{name}",\n      &{name}, {argnames},\n'
             f'      py::arg("stream"));\n}}')
    text = text.rstrip()
    assert text.endswith("}")
    text = text[:-1] + block
    path.write_text(text, encoding="utf-8")


def patch_thin(spec: dict) -> None:
    name = spec["python_name"]
    path = THIN / "_thin.py"
    text = path.read_text(encoding="utf-8")
    if f"def {name}(" in text:
        return
    py = spec.get("python", {})
    positional = py.get("positional", [])
    defaults = py.get("defaults", {})
    kw = [a["name"] for a in spec["args"]
          if a["name"] not in positional and a["kind"] != "out_tensor"]
    sig = ", ".join(positional)
    if kw:
        sig += ", *, " + ", ".join(
            f"{k}={defaults.get(k, 'None')}" for k in kw)
    lines = [f"\n\ndef {name}({sig}):",
             "    ext = _extension()"]
    for a in spec["args"]:
        if a["kind"] == "int_array":
            v = a["name"]
            lines.append(
                f"    {v} = [] if {v} is None else "
                f"[int(v) for v in {v}]")
    call_args = []
    for a in spec["args"]:
        if a["kind"] == "out_tensor":
            continue
        kind, v = a["kind"], a["name"]
        if kind == "tensor" or kind == "optional_tensor":
            call_args.append(v)
        elif kind == "int_array":
            call_args.append(v)
        elif kind == "char_ptr":
            call_args.append(f"str({v})")
        elif kind in ("int64", "bool", "double", "float"):
            cast = {"int64": "int", "bool": "bool",
                    "double": "float", "float": "float"}[kind]
            call_args.append(f"{cast}({v})")
    body = "\n".join(lines)
    body += "\n    return ext.%s(\n        %s,\n        _current_stream_ptr(),\n    )" % (
        name, ",\n        ".join(call_args))
    text = text.rstrip() + "\n" + body + "\n"
    path.write_text(text, encoding="utf-8")


def patch_whitelist(spec: dict) -> None:
    # Whitelist is derived dynamically from _thin module functions; no patch.
    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).parent))
    from op_spec_codegen import generate

    generated_path = CSRC / "ops_generated.cpp"
    text = generated_path.read_text(encoding="utf-8")
    if f"at::Tensor {spec['python_name']}(" not in text:
        with generated_path.open("a", encoding="utf-8") as fh:
            fh.write("\n// ============ generated from "
                     f"{spec['aclnn_name']} ============\n")
            fh.write(generate(spec))
            fh.write("\n")
    patch_pybind(spec)
    patch_thin(spec)
    patch_whitelist(spec)
    print(f"applied {spec['python_name']}: cpp/pybind/_thin/whitelist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
