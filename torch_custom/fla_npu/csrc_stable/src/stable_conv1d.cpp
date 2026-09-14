// Stable-ABI adapters for the causal conv1d family.
//
// aclnn exposes one entry point for all three public APIs; what differs is
// `runMode` (0 = prefill/FN, 1 = decode/UPDATE) plus how the caller's metadata
// is shaped.  The launcher therefore has one C++ implementation and three thin
// entry points:
//
//   * npu_causal_conv1d_fn      -- prefill; runMode 0, headNum forwarded, the
//                                  device tensor metadata plus its host twins.
//   * npu_causal_conv1d_update  -- decode; runMode 1, no padding (the reference
//                                  passes INT64_MIN for padSlotId).
//   * npu_causal_conv1d         -- deprecated compatibility API; runMode and
//                                  headNum come from the caller and every
//                                  metadata slot arrives as a host array.
//
// The scheduling parameters the reference refuses (block cache / APC /
// metadata objects) are refused by the Python wrappers, so nothing here has to
// describe them.
//
// Included by stable_ops.cpp (single TU); registration lives there.

#include "thin_stable/at_facade.h"
#include "thin_stable/boxed.h"
#include "thin_stable/exec.h"

#include <cstdint>
#include <optional>

namespace {

using torch::stable::Tensor;
using fla_npu_thin::stable::TensorMeta;
using fla_npu_thin::stable::allocate_like;
using fla_npu_thin::stable::allocate_sizes;
using fla_npu_thin::stable::cstr;
using fla_npu_thin::stable::int_array;
using fla_npu_thin::stable::meta_of;
using fla_npu_thin::stable::optional_tensor;
using fla_npu_thin::stable::out_tensor;
using fla_npu_thin::stable::scalar;
using fla_npu_thin::stable::size_of;
using fla_npu_thin::stable::tensor;

constexpr const char* kActivationNames[] = {"none", "silu", "swish"};

// maxQueryLen is only meaningful for the varlen update form; the other two
// entry points pass the reference's -1.
constexpr int64_t kNoQueryLenBound = -1;

// Mirrors _infer_causal_conv1d_y: the prefill path with a positive headNum
// returns the head-split view instead of a same-shape copy.
Tensor infer_out(const TensorMeta& x_meta, int64_t head_num, int64_t run_mode) {
  if (run_mode == 0 && head_num > 0) {
    if (x_meta.ndim == 3) {
      return allocate_sizes({size_of(x_meta, 0), head_num, size_of(x_meta, 1),
                             size_of(x_meta, 2) / head_num},
                            x_meta.scalar_type, x_meta);
    }
    if (x_meta.ndim == 2) {
      return allocate_sizes({head_num, size_of(x_meta, 0),
                             size_of(x_meta, 1) / head_num},
                            x_meta.scalar_type, x_meta);
    }
  }
  return allocate_like(x_meta);
}

// The single aclnn argument list, in the order the header declares it.
Tensor launch(Tensor x, Tensor weight, std::optional<Tensor> bias,
              std::optional<Tensor> conv_states,
              std::optional<Tensor> query_start_loc,
              std::optional<Tensor> cache_indices,
              std::optional<Tensor> has_initial_state,
              std::optional<Tensor> num_accepted_tokens,
              std::optional<Tensor> query_start_loc_cpu,
              std::optional<Tensor> cache_indices_cpu,
              std::optional<Tensor> has_initial_state_cpu,
              std::optional<Tensor> num_accepted_tokens_cpu, int64_t activation,
              int64_t pad_slot_id, int64_t null_block_id, int64_t run_mode,
              int64_t head_num, int64_t max_query_len, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  Tensor out = infer_out(x_meta, head_num, run_mode);
  FLA_STABLE_EXEC("aclnnCausalConv1d", x_meta, stream, tensor(x_meta),
                  tensor(meta_of(weight)), optional_tensor(bias),
                  optional_tensor(conv_states), optional_tensor(query_start_loc),
                  optional_tensor(cache_indices),
                  optional_tensor(has_initial_state),
                  optional_tensor(num_accepted_tokens),
                  int_array(query_start_loc_cpu),
                  int_array(cache_indices_cpu),
                  int_array(has_initial_state_cpu),
                  int_array(num_accepted_tokens_cpu),
                  cstr(kActivationNames, activation), scalar(pad_slot_id),
                  scalar(null_block_id), scalar(run_mode), scalar(head_num),
                  scalar(max_query_len), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_fn =
    "npu_causal_conv1d_fn(Tensor x, Tensor weight, Tensor? bias, "
    "Tensor? conv_states, Tensor? query_start_loc, Tensor? cache_indices, "
    "Tensor? has_initial_state, Tensor? query_start_loc_cpu, "
    "Tensor? cache_indices_cpu, Tensor? has_initial_state_cpu, int activation, "
    "int pad_slot_id, int null_block_id, int head_num, int stream) -> Tensor";

Tensor run_npu_causal_conv1d_fn(
    Tensor x, Tensor weight, std::optional<Tensor> bias,
    std::optional<Tensor> conv_states, std::optional<Tensor> query_start_loc,
    std::optional<Tensor> cache_indices,
    std::optional<Tensor> has_initial_state,
    std::optional<Tensor> query_start_loc_cpu,
    std::optional<Tensor> cache_indices_cpu,
    std::optional<Tensor> has_initial_state_cpu, int64_t activation,
    int64_t pad_slot_id, int64_t null_block_id, int64_t head_num,
    int64_t stream) {
  return launch(x, weight, bias, conv_states, query_start_loc, cache_indices,
                has_initial_state, /*num_accepted_tokens=*/std::nullopt,
                query_start_loc_cpu, cache_indices_cpu, has_initial_state_cpu,
                /*num_accepted_tokens_cpu=*/std::nullopt, activation,
                pad_slot_id, null_block_id, /*run_mode=*/0, head_num,
                /*max_query_len=*/kNoQueryLenBound, stream);
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_update =
    "npu_causal_conv1d_update(Tensor x, Tensor conv_state, Tensor weight, "
    "Tensor? bias, int activation, Tensor? conv_state_indices, "
    "Tensor? num_accepted_tokens, Tensor? query_start_loc, int max_query_len, "
    "int null_block_id, Tensor? conv_state_indices_cpu, "
    "Tensor? num_accepted_tokens_cpu, Tensor? query_start_loc_cpu, int stream) "
    "-> Tensor";

Tensor run_npu_causal_conv1d_update(
    Tensor x, Tensor conv_state, Tensor weight, std::optional<Tensor> bias,
    int64_t activation, std::optional<Tensor> conv_state_indices,
    std::optional<Tensor> num_accepted_tokens,
    std::optional<Tensor> query_start_loc, int64_t max_query_len,
    int64_t null_block_id, std::optional<Tensor> conv_state_indices_cpu,
    std::optional<Tensor> num_accepted_tokens_cpu,
    std::optional<Tensor> query_start_loc_cpu, int64_t stream) {
  // UPDATE never pads, which the reference spells as INT64_MIN rather than -1.
  constexpr int64_t kNoPadding = -9223372036854775807LL - 1;
  return launch(x, weight, bias, conv_state, query_start_loc,
                conv_state_indices, /*has_initial_state=*/std::nullopt,
                num_accepted_tokens, query_start_loc_cpu,
                conv_state_indices_cpu, /*has_initial_state_cpu=*/std::nullopt,
                num_accepted_tokens_cpu, activation, kNoPadding, null_block_id,
                /*run_mode=*/1, /*head_num=*/0, max_query_len, stream);
}

// ---------------------------------------------------------------------------
// deprecated host-metadata compatibility API
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d =
    "npu_causal_conv1d(Tensor x, Tensor weight, Tensor? bias, "
    "Tensor? conv_states, Tensor? query_start_loc, Tensor? cache_indices, "
    "Tensor? initial_state_mode, Tensor? num_accepted_tokens, int activation, "
    "int pad_slot_id, int run_mode, int head_num, int stream) -> Tensor";

Tensor run_npu_causal_conv1d(
    Tensor x, Tensor weight, std::optional<Tensor> bias,
    std::optional<Tensor> conv_states,
    std::optional<Tensor> query_start_loc,
    std::optional<Tensor> cache_indices,
    std::optional<Tensor> initial_state_mode,
    std::optional<Tensor> num_accepted_tokens, int64_t activation,
    int64_t pad_slot_id, int64_t run_mode, int64_t head_num, int64_t stream) {
  return launch(x, weight, bias, conv_states, /*query_start_loc=*/std::nullopt,
                /*cache_indices=*/std::nullopt,
                /*has_initial_state=*/std::nullopt,
                /*num_accepted_tokens=*/std::nullopt, query_start_loc,
                cache_indices, initial_state_mode, num_accepted_tokens,
                activation, pad_slot_id, /*null_block_id=*/-1, run_mode, head_num,
                /*max_query_len=*/kNoQueryLenBound, stream);
}

}  // namespace
