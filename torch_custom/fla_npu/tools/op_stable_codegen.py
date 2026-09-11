#!/usr/bin/env python3
"""Generate Stable-ABI adapters from the existing op specs.

The pybind codegen (op_spec_codegen.py) turned the specs into adapters that
speak ATen C++.  This backend emits the same operators against the stable C
shims, so a spec stays the single source of truth for both.

What is supported here (everything else is reported as "manual"):
  * kinds: tensor, optional_tensor, int_array (carried as a host int64 tensor,
    see acl_meta.h), int64, bool, double, float
  * structured outputs (``output`` / ``outputs`` with source/dtype/shape/when)
  * cpp_only args feeding output conditions

Not supported (needs a per-op adapter):
  * ``char_ptr`` args: the stable conversions have no std::string, so a layout
    string must be encoded as an int by hand
  * raw ``alloc`` expressions: they are written against ATen idioms

Usage:
  python op_stable_codegen.py --all            # write the generated aggregate
  python op_stable_codegen.py --parse-only     # report per-spec support
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SPEC_DIR = Path(__file__).resolve().parents[1] / "op_specs"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "csrc_stable" / "generated" / "ops_stable_generated.inc"

_DTYPE_ID = {"float32": 6, "bfloat16": 15, "float16": 5, "int32": 3,
             "int64": 4, "uint8": 0, "bool": 11}

_ACLNN_TYPE = {
    "tensor": "const aclTensor*",
    "optional_tensor": "const aclTensor*",
    "out_tensor": "aclTensor*",
    "int_array": "const aclIntArray*",
    "int64": "int64_t",
    "bool": "bool",
    "double": "double",
    "float": "float",
    "char_ptr": "const char*",
}

# Adapters that already exist by hand (they need the ownership/shape handling
# that the generator does not express yet).
HAND_WRITTEN = {"npu_recurrent_gated_delta_rule", "npu_recurrent_kda"}


def unsupported_reasons(spec: dict) -> list[str]:
    reasons: list[str] = []
    if not spec.get("enabled", True):
        reasons.append("spec disabled")
    for argument in spec["args"]:
        if argument["kind"] == "char_ptr":
            if not argument.get("enum"):
                reasons.append(
                    f"char_ptr arg {argument['name']} has no enum table")
        if argument["kind"] not in (*_ACLNN_TYPE, "char_ptr"):
            reasons.append(f"unsupported kind {argument['kind']}")
    outputs = spec.get("outputs") or [spec.get("output") or {}]
    for entry in outputs:
        if "alloc" in entry and "at_shim" not in entry.get("alloc", ""):
            # Alloc expressions are supported through the ATen-shaped facade
            # (allocs are spliced verbatim); only idioms the facade lacks would
            # need a manual adapter, and those show up as compile errors.
            pass
        if entry.get("dtype") not in (None, "same", "source", *_DTYPE_ID,
                                      "output_dtype"):
            reasons.append(f"unsupported output dtype {entry.get('dtype')!r}")
    return reasons


def _arg_cpp(argument: dict) -> str:
    return argument.get("cpp", argument["name"])


def _shim(text: str) -> str:
    """Rewrite the spec's ATen idioms onto the facade alias.

    The alias cannot be called ``at``: torch's own headers (pulled in by
    torch/csrc/stable) declare a real ``at`` namespace, and ``at::Tensor`` would
    then be ambiguous.  ``shim::`` is unambiguous and keeps the spec text
    otherwise verbatim.
    """

    return text.replace("at::", "shim::")


def _cpp_params(spec: dict) -> list[tuple[str, str, bool]]:
    """(name, kind, cpp_only) for every non-output arg, in spec order."""

    return [(_arg_cpp(a), a["kind"], bool(a.get("cpp_only")))
            for a in spec["args"] if a["kind"] != "out_tensor"]


def _schema(spec: dict) -> str:
    parts = []
    for argument in spec["args"]:
        if argument["kind"] == "out_tensor":
            continue
        name = argument["name"]
        kind = argument["kind"]
        if kind == "tensor":
            parts.append(f"Tensor {name}")
        elif kind == "optional_tensor":
            parts.append(f"Tensor? {name}")
        elif kind == "int_array":
            parts.append(f"Tensor? {name}")
        elif kind == "bool":
            parts.append(f"bool {name}")
        elif kind == "int64":
            parts.append(f"int {name}")
        elif kind in ("double", "float"):
            parts.append(f"float {name}")
        elif kind == "char_ptr":
            # Encoded as an int: the stable conversions carry no strings.
            parts.append(f"int {name}")
    # The raw stream is an explicit argument: the stable accelerator stream API
    # returns 0 on torch_npu, so the Python wrapper forwards the caller's stream.
    parts.append("int stream")
    outputs = [a for a in spec["args"] if a["kind"] == "out_tensor"]
    output_spec = spec.get("outputs") or [spec.get("output") or {}]
    returns = []
    for index, _ in enumerate(outputs):
        entry = output_spec[index] if index < len(output_spec) else {}
        when = entry.get("when")
        always_null = entry.get("alloc", "").strip() == "at::Tensor()"
        returns.append("Tensor?" if (when or always_null) else "Tensor")
    ret = returns[0] if len(returns) == 1 else "(" + ", ".join(returns) + ")"
    return f"{spec['python_name']}({', '.join(parts)}) -> {ret}"


def optional_output_mask(spec: dict) -> list[bool]:
    """Whether each out_tensor slot is declared ``Tensor?`` in the schema."""

    outputs = [a for a in spec["args"] if a["kind"] == "out_tensor"]
    entries = spec.get("outputs") or [spec.get("output") or {}]
    mask = []
    for index in range(len(outputs)):
        entry = entries[index] if index < len(entries) else {}
        mask.append(bool(entry.get("when"))
                    or entry.get("alloc", "").strip() == "shim::Tensor()"
                    or entry.get("alloc", "").strip() == "at::Tensor()")
    return mask


def _size_exprs(entry: dict, arg_names: set[str]) -> str | None:
    shape = entry.get("shape")
    if shape is None:
        return None
    parts = []
    for item in shape:
        if "dim" in item:
            parts.append(f"size_of({item['arg']}_meta, {item['dim']})")
        elif "arg" in item:
            # A bare arg name in a size list is a scalar local (chunk_size ...).
            parts.append(item["arg"])
        elif "expr" in item:
            parts.append(item["expr"])
        else:
            raise ValueError(f"bad shape item {item!r}")
    return "{" + ", ".join(parts) + "}"


def _alloc_text(entry: dict, default_source: str, arg_names: set[str]) -> str:
    """Structured output description -> facade alloc expression.

    Used for specs that mix raw ``alloc`` outputs with structured ones, so the
    whole op can be emitted through the single facade path.
    """

    source = entry.get("source", default_source)
    dtype = entry.get("dtype", "float32")
    sizes = _size_exprs(entry, arg_names)
    if dtype in ("same", "source") or dtype == "output_dtype":
        options = f"{source}.options()"
    elif dtype in _DTYPE_ID:
        aten = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16",
                "float16": "at::kHalf", "int32": "at::kInt",
                "int64": "at::kLong", "uint8": "at::kByte",
                "bool": "at::kBool"}[dtype]
        options = f"{source}.options().dtype({aten})"
    else:
        raise ValueError(f"unsupported output dtype {dtype!r}")
    if dtype == "output_dtype":
        # dtype follows the char_ptr arg; express it with the facade's constants.
        options = (f"({source}.options().dtype("
                   f'output_dtype == "float32" ? at::kFloat : at::kBFloat16))')
    if sizes is None:
        return f"at::empty_like({source})" if dtype in ("same", "source") \
            else f"at::empty({{{source}.sizes()}}, {options})"
    return f"at::empty({sizes}, {options})"


def generate_cpp(spec: dict) -> str:
    name = spec["python_name"]
    outputs = [a for a in spec["args"] if a["kind"] == "out_tensor"]
    output_spec = spec.get("outputs") or [spec.get("output") or {}]
    params = _cpp_params(spec)
    arg_names = {p[0] for p in params}
    aclnn_args = [a for a in spec["args"] if not a.get("cpp_only")]
    uses_alloc = any("alloc" in entry for entry in output_spec)
    optional_mask = optional_output_mask(spec)
    lines: list[str] = []
    lines.append(f"// ---- generated: {name} ----")
    for argument in spec["args"]:
        if argument["kind"] == "char_ptr":
            arg = _arg_cpp(argument)
            lines.append(f"inline const char* {name}_{arg}_name(int64_t code) {{")
            lines.append("  switch (code) {")
            for code, value in enumerate(argument["enum"]):
                lines.append(f'    case {code}: return "{value}";')
            lines.append("    default:")
            lines.append("      throw std::runtime_error(")
            lines.append(
                f'          "{name}: bad {arg} code " + std::to_string(code));')
            lines.append("  }")
            lines.append("}")
    lines.append(f"using {name}_GetWorkspaceFn = int (*)(")
    lines.append("    " + ",\n    ".join(
        _ACLNN_TYPE[a["kind"]] for a in aclnn_args) + ",")
    lines.append("    uint64_t*, aclOpExecutor**);")
    lines.append(f"constexpr const char* kSchema_{name} =")
    lines.append(f'    "{_schema(spec)}";')
    lines.append("")
    lines.append(f"void boxed_{name}(StableIValue* stack, uint64_t num_inputs,")
    lines.append("                 uint64_t num_outputs) {")
    lines.append("  (void)num_inputs;")
    lines.append("  (void)num_outputs;")
    index = 0
    for cpp_name, kind, cpp_only in params:
        if cpp_only:
            # Conditions such as `use_beta_sigmoid_in_kernel` drive output
            # allocation; keep them as plain locals.
            if kind == "bool":
                lines.append(f"  const bool {cpp_name} = to<bool>(stack[{index}]);")
            else:
                lines.append(
                    f"  const {kind} {cpp_name} = to<int64_t>(stack[{index}]);")
            index += 1
            continue
        if kind == "tensor":
            lines.append(f"  const Tensor {cpp_name}_t = to<Tensor>(stack[{index}]);")
            lines.append(f"  const TensorMeta {cpp_name}_meta = meta_of({cpp_name}_t);")
            if uses_alloc:
                # Alloc expressions reference args by name (v.size(2), ...).
                lines.append(
                    f"  const shim::Tensor {cpp_name}({cpp_name}_t);")
        elif kind == "optional_tensor":
            lines.append(
                f"  const auto {cpp_name}_t = to<std::optional<Tensor>>(stack[{index}]);")
            lines.append(
                f"  const TensorMeta {cpp_name}_meta = "
                f"{cpp_name}_t.has_value() ? meta_of(*{cpp_name}_t) : TensorMeta();")
            if uses_alloc:
                lines.append(
                    f"  const std::optional<shim::Tensor> {cpp_name} = "
                    f"{cpp_name}_t.has_value() "
                    f"? std::optional<shim::Tensor>(shim::Tensor(*{cpp_name}_t)) "
                    f": std::nullopt;")
        elif kind == "int_array":
            lines.append(
                f"  const auto {cpp_name}_t = to<std::optional<Tensor>>(stack[{index}]);")
            # Alloc text written against the pybind backend treats an int_array
            # arg as std::vector<int64_t> (query_start_loc.empty(), .size()), so
            # the values keep the arg's own name there.
            values_name = cpp_name if uses_alloc else f"{cpp_name}_values"
            lines.append(
                f"  const std::vector<int64_t> {values_name} = "
                f"{cpp_name}_t.has_value() ? host_int_values({cpp_name}_t->get()) "
                f": std::vector<int64_t>();")
            lines.append(f"  AclIntArrayView {cpp_name}_view({values_name});")
        elif kind == "bool":
            lines.append(f"  const bool {cpp_name} = to<bool>(stack[{index}]);")
        elif kind == "int64":
            lines.append(f"  const int64_t {cpp_name} = to<int64_t>(stack[{index}]);")
        elif kind in ("double", "float"):
            lines.append(f"  const double {cpp_name} = to<double>(stack[{index}]);")
        elif kind == "char_ptr":
            if uses_alloc:
                # Alloc/comparison text uses the string form, so keep the code
                # under a separate name.
                lines.append(
                    f"  const int64_t {cpp_name}_code = to<int64_t>(stack[{index}]);")
                lines.append(
                    f"  const std::string {cpp_name} = "
                    f"{name}_{cpp_name}_name({cpp_name}_code);")
            else:
                lines.append(
                    f"  const int64_t {cpp_name} = to<int64_t>(stack[{index}]);")
        index += 1
    lines.append(f"  const int64_t stream = to<int64_t>(stack[{index}]);")
    lines.append("")
    lines.append("  auto& rt = Runtime::instance();")
    lines.append(f"  auto get_ws = reinterpret_cast<{name}_GetWorkspaceFn>(")
    lines.append(f'      rt.symbol("{spec["aclnn_name"]}GetWorkspaceSize"));')
    lines.append("  auto launch = reinterpret_cast<LaunchFn>(")
    lines.append(f'      rt.symbol("{spec["aclnn_name"]}"));')
    lines.append("  uint64_t workspace_size = 0;")
    lines.append("  aclOpExecutor* executor = nullptr;")
    lines.append("")
    if uses_alloc:
        lines.append("  std::vector<shim::Tensor> outputs;")
    else:
        lines.append("  std::vector<Tensor> outputs;")
        lines.append("  std::vector<bool> output_present;")
    for position, entry in enumerate(output_spec):
        when = entry.get("when")
        if uses_alloc:
            # Splice the spec's alloc text verbatim; the facade supplies the
            # ATen shape it is written against.
            text = entry.get("alloc")
            if text is None:
                text = _alloc_text(entry, params[0][0], arg_names)
            text = _shim(text)
            if when:
                lines.append(f"  if ({when}) {{")
                lines.append(f"    outputs.push_back({text});")
                lines.append("  } else {")
                lines.append("    outputs.push_back(shim::Tensor());")
                lines.append("  }")
            else:
                lines.append(f"  outputs.push_back({text});")
            continue
        source = entry.get("source", params[0][0])
        dtype = entry.get("dtype", "float32")
        sizes = _size_exprs(entry, arg_names)
        if dtype in ("same", "source"):
            dtype_expr = f"{source}_meta.scalar_type"
            if sizes is None:
                alloc = f"allocate_like({source}_meta)"
            else:
                alloc = (f"allocate_sizes({sizes}, {dtype_expr}, "
                         f"{source}_meta)")
        elif dtype == "output_dtype":
            # dtype follows the char_ptr arg of the same name; the pybind codegen
            # spells it as `output_dtype == "float32" ? kFloat : kBFloat16`.
            enum_arg = next(a for a in spec["args"]
                            if a["name"] == "output_dtype")
            ids = [_DTYPE_ID[value] for value in enum_arg["enum"]]
            expr = str(ids[-1])
            for code in range(len(ids) - 2, -1, -1):
                expr = f"(output_dtype == {code} ? {ids[code]} : {expr})"
            alloc = (f"allocate_sizes({sizes}, {expr}, {source}_meta)"
                     if sizes is not None
                     else f"allocate_sizes({{{source}_meta.sizes}}, {expr}, "
                          f"{source}_meta)")
        elif dtype in _DTYPE_ID:
            dtype_expr = str(_DTYPE_ID[dtype])
            alloc = (f"allocate_sizes({sizes}, {dtype_expr}, {source}_meta)"
                     if sizes is not None
                     else f"allocate_sizes({{{source}_meta.sizes}}, "
                          f"{dtype_expr}, {source}_meta)")
        else:
            raise ValueError(f"{name}: unsupported output dtype {dtype!r}")
        if when:
            lines.append(f"  if ({when}) {{")
            lines.append(f"    outputs.push_back({alloc});")
            lines.append("    output_present.push_back(true);")
            lines.append("  } else {")
            lines.append("    outputs.push_back(Tensor());")
            lines.append("    output_present.push_back(false);")
            lines.append("  }")
        else:
            lines.append(f"  outputs.push_back({alloc});")
            lines.append("  output_present.push_back(true);")
    lines.append("")
    lines.append("  std::vector<std::unique_ptr<AclTensorView>> views;")
    out_index = 0
    for cpp_name, kind, cpp_only in params:
        if cpp_only:
            continue
        if kind == "tensor":
            lines.append(
                f"  views.push_back(std::make_unique<AclTensorView>({cpp_name}_meta));")
        elif kind == "optional_tensor":
            lines.append(
                f"  views.push_back(std::make_unique<AclTensorView>({cpp_name}_meta));")
    for position, _ in enumerate(outputs):
        if uses_alloc:
            # An undefined output (at::Tensor()) must stay a null descriptor,
            # exactly like the pybind path passing at::Tensor() through.
            lines.append(
                f"  views.push_back(std::make_unique<AclTensorView>("
                f"outputs[{position}].defined() "
                f"? meta_of(outputs[{position}].tensor()) : TensorMeta()));")
        else:
            lines.append(
                f"  views.push_back(std::make_unique<AclTensorView>("
                f"meta_of(outputs[{position}])));")
        out_index += 1
    lines.append("")
    tokens = []
    view_index = 0
    for cpp_name, kind, cpp_only in params:
        if cpp_only:
            continue
        if kind in ("tensor", "optional_tensor"):
            tokens.append(f"views[{view_index}]->get()")
            view_index += 1
        elif kind == "int_array":
            tokens.append(f"{cpp_name}_view.get()")
        elif kind == "char_ptr":
            tokens.append(f"{cpp_name}.c_str()" if uses_alloc
                          else f"{name}_{cpp_name}_name({cpp_name})")
        else:
            tokens.append(cpp_name)
    for position in range(len(outputs)):
        tokens.append(f"views[{view_index}]->get()")
        view_index += 1
    call = f"get_ws(\n      " + ",\n      ".join(tokens) + ",\n      &workspace_size, &executor)"
    lines.append(f"  const int get_ret = {call};")
    lines.append("  if (get_ret != 0) {")
    lines.append("    throw std::runtime_error(")
    lines.append(f'        "{spec["aclnn_name"]}GetWorkspaceSize failed: " +')
    lines.append("        std::to_string(get_ret));")
    lines.append("  }")
    lines.append("  Tensor workspace;")
    lines.append("  void* workspace_ptr = nullptr;")
    lines.append("  if (workspace_size != 0) {")
    lines.append("    workspace = allocate_bytes(static_cast<int64_t>(workspace_size),")
    lines.append(f"                                   {params[0][0]}_meta);")
    lines.append("    TORCH_ERROR_CODE_CHECK(")
    lines.append("        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));")
    lines.append("  }")
    lines.append("  const int launch_ret = launch(workspace_ptr, workspace_size,")
    lines.append("                                executor,")
    lines.append("                                reinterpret_cast<void*>(stream));")
    lines.append("  if (launch_ret != 0) {")
    lines.append("    throw std::runtime_error(")
    lines.append(f'        "{spec["aclnn_name"]} failed: " +')
    lines.append("        std::to_string(launch_ret));")
    lines.append("  }")
    for position in range(len(outputs)):
        slot_is_optional = optional_mask[position]
        value = (f"outputs[{position}].tensor()" if uses_alloc
                 else f"outputs[{position}]")
        present = (f"outputs[{position}].defined()" if uses_alloc
                   else f"output_present[{position}]")
        if not uses_alloc:
            # Track presence explicitly: Tensor::defined() pulls in the 2.9-only
            # aoti_torch_is_defined symbol, which we must not depend on.
            pass
        if slot_is_optional:
            # An optional slot's StableIValue is a *pointer to a heap
            # StableIValue* (see FromImpl<std::optional<T>>), not a bare handle:
            # packing from(Tensor) here makes the dispatcher dereference a
            # tensor handle as a pointer (SIGSEGV inside to_ivalue).
            lines.append(
                f"  stack[{position}] = {present} "
                f"? from(std::optional<Tensor>({value})) : from(std::nullopt);")
        else:
            lines.append(f"  stack[{position}] = from({value});")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def schema_order(spec: dict) -> list[tuple[str, str]]:
    """(arg name, kind) in the order the dispatcher op expects."""

    return [(a["name"], a["kind"]) for a in spec["args"]
            if a["kind"] != "out_tensor"]


def enum_tables(spec: dict) -> dict[str, dict[str, int]]:
    return {a["name"]: {value: code for code, value in enumerate(a["enum"])}
            for a in spec["args"]
            if a["kind"] == "char_ptr" and a.get("enum")}


def output_rules(spec: dict) -> list[tuple[int, str | None]]:
    """(output position, when-expression or None) for each out_tensor."""

    outputs = [a for a in spec["args"] if a["kind"] == "out_tensor"]
    entries = spec.get("outputs") or [spec.get("output") or {}]
    return [(index, (entries[index] if index < len(entries) else {}).get("when"))
            for index in range(len(outputs))]


def generate_python(spec: dict) -> str:
    name = spec["python_name"]
    py = spec.get("python", {})
    positional = py.get("positional", [])
    defaults = py.get("defaults", {})
    ignored = py.get("ignored", [])
    kw = [a["name"] for a in spec["args"]
          if a["name"] not in positional and a["kind"] != "out_tensor"]
    kw = [k for k in kw if k not in ignored]
    sig_kw = kw + [k for k in ignored if k not in kw]
    signature = ", ".join(positional)
    if sig_kw:
        separator = "*, " if not positional else ", *, "
        signature += separator + ", ".join(
            f"{key}={defaults.get(key, 'None')}" for key in sig_kw)
    lines = [f"def {name}({signature}):"]
    # The same pre-block the pybind wrapper gets: default resolution (scale,
    # lower_bound, layout) and explicit ctypes fallbacks for sub-domains the
    # kernel itself does not support.
    pre = py.get("pre")
    if pre:
        for raw_line in pre.splitlines():
            lines.append(("    " + raw_line) if raw_line.strip() else "")
    if py.get("derive_chunk_indices") and "chunk_size" in {
            a["name"] for a in spec["args"]}:
        lines.append("    if cu_seqlens and not chunk_indices:")
        lines.append("        chunk_indices = []")
        lines.append("        for _seq in range(len(cu_seqlens) - 1):")
        lines.append("            _len = cu_seqlens[_seq + 1] - cu_seqlens[_seq]")
        lines.append("            for _c in range((_len + chunk_size - 1) // chunk_size):")
        lines.append("                chunk_indices.extend((_seq, _c))")
    target = f'_call("{name}", {{'
    if py.get("return_code"):
        # Custom return shapes (e.g. kda_fwd's (*outputs, initial_state)) are
        # expressed against `result`, the same convention as the pybind wrapper.
        lines.append(f"    result = {target}")
    else:
        lines.append(f"    return {target}")
    for argument in spec["args"]:
        if argument["kind"] == "out_tensor":
            continue
        lines.append(f'        "{argument["name"]}": {argument["name"]},')
    lines.append("    })")
    if py.get("return_code"):
        for raw_line in py["return_code"].splitlines():
            lines.append(("    " + raw_line) if raw_line.strip() else "")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    supported, unsupported = [], []
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        reasons = unsupported_reasons(spec)
        if spec["python_name"] in HAND_WRITTEN:
            reasons.append("hand-written adapter already registered")
        (unsupported if reasons else supported).append((spec, reasons, path))

    for spec, reasons, path in unsupported:
        print(f"MANUAL {spec['python_name']}: {'; '.join(sorted(set(reasons)))}")
    for spec, _, path in supported:
        print(f"GEN    {spec['python_name']}")
    print(f"\n{supported and len(supported)} generatable, "
          f"{len(unsupported)} need a manual adapter")
    if args.parse_only or not args.all:
        return 0
    body = []
    for spec, _, _ in supported:
        body.append(generate_cpp(spec))
    header = [
        "// Generated by tools/op_stable_codegen.py -- do not edit by hand.",
        "// Included by csrc_stable/src/stable_ops.cpp (single TU).",
        '#include "thin_stable/at_facade.h"',
        "#include <memory>",
        "#include <optional>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace {",
        "using torch::stable::Tensor;",
        "using fla_npu_thin::Runtime;",
        "using fla_npu_thin::stable::at_shim::TensorOptions;",
        "using fla_npu_thin::stable::at_shim::empty;",
        "using fla_npu_thin::stable::at_shim::empty_like;",
        "using fla_npu_thin::stable::at_shim::kBFloat16;",
        "using fla_npu_thin::stable::at_shim::kFloat;",
        "using fla_npu_thin::stable::at_shim::kHalf;",
        "using fla_npu_thin::stable::AclIntArrayView;",
        "using fla_npu_thin::stable::AclTensorView;",
        "using fla_npu_thin::stable::LaunchFn;",
        "using fla_npu_thin::stable::TensorMeta;",
        "using fla_npu_thin::stable::aclIntArray;",
        "using fla_npu_thin::stable::aclOpExecutor;",
        "using fla_npu_thin::stable::aclTensor;",
        "using fla_npu_thin::stable::allocate_bytes;",
        "using fla_npu_thin::stable::allocate_like;",
        "using fla_npu_thin::stable::allocate_sizes;",
        "using fla_npu_thin::stable::host_int_values;",
        "using fla_npu_thin::stable::meta_of;",
        "using fla_npu_thin::stable::meta_of_handle;",
        "using fla_npu_thin::stable::size_of;",
        "",
        "// Spec alloc expressions are written as at::empty(...)/at::Tensor();",
        "// the facade supplies exactly that surface on top of the stable shims.",
        "namespace shim = fla_npu_thin::stable::at_shim;",
        "}  // namespace",
        "",
    ]
    # Spec helpers are written against the same ATen subset; emit each unique
    # block once (they are pure functions over at::Tensor views).
    helpers = []
    seen_helpers: set[str] = set()
    for spec, _, _ in supported:
        text = (spec.get("helpers") or "").strip()
        if text and text not in seen_helpers:
            seen_helpers.add(text)
            helpers.append(text)
    if helpers:
        header.append("namespace {")
        header.extend(_shim(text) for text in helpers)
        header.append("}  // namespace")
        header.append("")
    footer = ["""
namespace {
// Registration hooks called from stable_ops.cpp's single def/impl blocks.
template <typename Library>
void register_generated_defs(Library& m) {
"""]
    for spec, _, _ in supported:
        footer.append(f"  m.def(kSchema_{spec['python_name']});")
    footer.append("}\n")
    footer.append("template <typename Library>")
    footer.append("void register_generated_impls(Library& m) {")
    for spec, _, _ in supported:
        footer.append(
            f'  m.impl("{spec["python_name"]}", &boxed_{spec["python_name"]});')
    footer.append("}\n}  // namespace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + body + footer) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out} ({len(supported)} adapters)")

    # Python glue: same user-facing signatures as the pybind wrappers, with the
    # int[]/char_ptr encodings the stable ABI needs.
    py_lines = [
        '"""Generated by tools/op_stable_codegen.py -- do not edit by hand."""',
        "from ._stable import _call  # noqa: F401",
        "",
        "_SIG = {",
    ]
    for spec, _, _ in supported:
        name = spec["python_name"]
        entries = ", ".join(f'("{a}", "{k}")' for a, k in schema_order(spec))
        py_lines.append(f'    "{name}": [{entries}],')
    py_lines.append("}")
    py_lines.append("")
    py_lines.append("_ENUM = {")
    for spec, _, _ in supported:
        tables = enum_tables(spec)
        if not tables:
            continue
        inner = ", ".join(
            f'"{arg}": {{{", ".join(f"{v!r}: {c}" for v, c in table.items())}}}'
            for arg, table in tables.items())
        py_lines.append(f'    "{spec["python_name"]}": {{{inner}}},')
    py_lines.append("}")
    py_lines.append("")
    py_lines.append("_RET = {")
    for spec, _, _ in supported:
        rules = output_rules(spec)
        parts = []
        for index, when in rules:
            parts.append(f"({index}, {when!r})")
        py_lines.append(f'    "{spec["python_name"]}": [{", ".join(parts)}],')
    py_lines.append("}")
    py_lines.append("")
    for spec, _, _ in supported:
        py_lines.append(generate_python(spec))
        py_lines.append("")
    py_lines.append("__all__ = [")
    for spec, _, _ in supported:
        py_lines.append(f'    "{spec["python_name"]}",')
    py_lines.append("]")
    py_path = args.out.parents[2] / "fla_npu" / "ops" / "ascendc" / (
        "_stable_generated.py")
    py_path.write_text("\n".join(py_lines) + "\n", encoding="utf-8")
    print(f"wrote {py_path} ({len(supported)} wrappers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
