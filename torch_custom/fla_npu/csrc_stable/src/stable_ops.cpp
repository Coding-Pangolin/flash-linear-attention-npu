// Single translation unit for every Stable-ABI thin adapter.
//
// torch/csrc/stable/tensor_inl.h defines non-inline member functions (e.g.
// `Tensor::scalar_type()`), so including the stable headers from more than one
// TU fails at link time with "multiple definition of
// torch::stable::Tensor::scalar_type() const".  All adapters therefore live in
// this one file -- the same aggregation the pybind codegen already does with
// ops_generated.cpp.
#include "stable_recurrent_gdr.cpp"
#include "stable_recurrent_kda.cpp"
#include "../generated/ops_stable_generated.inc"

// Exactly one library-definition block and one implementation block per
// namespace per TU: the macros expand to a fixed static-init symbol name, so a
// second block for the same namespace would be a redefinition.  The codegen
// phase therefore collects every adapter's schema/impl into these two lists.
STABLE_TORCH_LIBRARY(fla_npu_thin, m) {
  m.def(kSchemaRecurrentGdr);
  m.def(kSchemaRecurrentKda);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.def("_stream_probe(int device_index) -> (int, int)");
#endif
  register_generated_defs(m);
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_thin, CompositeExplicitAutograd, m) {
  m.impl("npu_recurrent_gated_delta_rule", &boxed_recurrent_gated_delta_rule);
  m.impl("npu_recurrent_kda", &boxed_recurrent_kda);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.impl("_stream_probe", &boxed_stream_probe);
#endif
  register_generated_impls(m);
}
