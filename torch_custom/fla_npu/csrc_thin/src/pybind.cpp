#include <torch/extension.h>

#include "thin_launcher/runtime.h"

namespace fla_npu_thin {

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    double scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    uint64_t stream);

at::Tensor npu_causal_conv1d(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& conv_states,
    const std::vector<int64_t>& query_start_loc,
    const std::vector<int64_t>& cache_indices,
    const std::vector<int64_t>& initial_state_mode,
    const std::vector<int64_t>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode,
    int64_t head_num,
    uint64_t stream);

at::Tensor npu_kda_gate_cumsum(
    const at::Tensor& g,
    const c10::optional<at::Tensor>& A_log,
    const c10::optional<at::Tensor>& dt_bias,
    const std::vector<int64_t>& cu_seqlens,
    int64_t chunk_size,
    bool use_gate_in_kernel,
    bool safe_gate,
    double lower_bound,
    uint64_t stream);


at::Tensor npu_chunk_local_cumsum(
    const at::Tensor& g,
    const std::vector<int64_t>& cu_seqlens,
    const std::vector<int64_t>& chunk_indices,
    int64_t chunk_size,
    bool reverse,
    double scale,
    bool head_first,
    const std::string& output_dtype,
    uint64_t stream);

}  // namespace fla_npu_thin

PYBIND11_MODULE(_C_thin, m) {
  using namespace fla_npu_thin;
  m.def("init", [](const std::string& path) {
    Runtime::instance().init(path);
  });
  m.def(
      "npu_recurrent_gated_delta_rule",
      &npu_recurrent_gated_delta_rule, py::arg("query"), py::arg("key"),
      py::arg("value"), py::arg("state"), py::arg("beta"),
      py::arg("scale"), py::arg("actual_seq_lengths"),
      py::arg("ssm_state_indices"), py::arg("num_accepted_tokens"),
      py::arg("g"), py::arg("gk"), py::arg("stream"));
  m.def(
      "npu_causal_conv1d",
      &npu_causal_conv1d, py::arg("x"), py::arg("weight"),
      py::arg("bias"), py::arg("conv_states"),
      py::arg("query_start_loc"), py::arg("cache_indices"),
      py::arg("initial_state_mode"), py::arg("num_accepted_tokens"),
      py::arg("activation_mode"), py::arg("pad_slot_id"),
      py::arg("run_mode"), py::arg("head_num"),
      py::arg("stream"));
  m.def(
      "npu_kda_gate_cumsum",
      &npu_kda_gate_cumsum, py::arg("g"), py::arg("A_log"),
      py::arg("dt_bias"), py::arg("cu_seqlens"),
      py::arg("chunk_size"), py::arg("use_gate_in_kernel"),
      py::arg("safe_gate"), py::arg("lower_bound"), py::arg("stream"));
  m.def(
      "npu_chunk_local_cumsum",
      &npu_chunk_local_cumsum, py::arg("g"),
      py::arg("cu_seqlens"),
      py::arg("chunk_indices"),
      py::arg("chunk_size"),
      py::arg("reverse"),
      py::arg("scale"),
      py::arg("head_first"),
      py::arg("output_dtype"),
      py::arg("stream"));
}