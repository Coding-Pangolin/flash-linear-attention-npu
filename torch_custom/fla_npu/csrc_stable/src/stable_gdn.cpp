// Stable-ABI adapters: npu_causal_conv1d_bwd, npu_chunk_fwd_o,
// npu_chunk_gdn_bwd_intra.
//
// All three allocate several outputs whose shapes come from more than one
// input, which is the only thing that separates them from the batch in
// stable_chunk.cpp:
//
//   * causal_conv1d_bwd sizes its d(initial_state) output from the segment
//     count, so the query_start_loc values are taken once with `int_values`
//     and the same vector is handed to the call.
//   * chunk_fwd_o's output shape depends on the layout string, so the enum name
//     is resolved once and used both for the allocation and for aclnn.
//   * chunk_gdn_bwd_intra mixes an output shaped from q and v with two that
//     copy v.
//
// Included by stable_ops.cpp (single TU); registration lives there.

#include "thin_stable/at_facade.h"
#include "thin_stable/boxed.h"
#include "thin_stable/exec.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_thin::stable::CStrArg;
using fla_npu_thin::stable::TensorMeta;
using fla_npu_thin::stable::allocate_like;
using fla_npu_thin::stable::allocate_sizes;
using fla_npu_thin::stable::enum_name;
using fla_npu_thin::stable::int_array;
using fla_npu_thin::stable::int_values;
using fla_npu_thin::stable::meta_of;
using fla_npu_thin::stable::optional_tensor;
using fla_npu_thin::stable::out_tensor;
using fla_npu_thin::stable::scalar;
using fla_npu_thin::stable::size_of;
using fla_npu_thin::stable::tensor;

// Table order is the code order; _stable.py's _char_code tables must match.
constexpr const char* kCausalConv1dBwdInputLayoutNames[] = {"BSND", "BNSD",
                                                            "NTD", "TND"};
constexpr const char* kChunkFwdOOutputLayoutNames[] = {"BNSD", "BSND", "TND",
                                                       "NTD"};

// ---------------------------------------------------------------------------
// npu_causal_conv1d_bwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_causal_conv1d_bwd =
    "npu_causal_conv1d_bwd(Tensor x, Tensor? y, Tensor weight, Tensor dy, "
    "Tensor? initial_state, Tensor? dht, Tensor? query_start_loc, "
    "int activation, int input_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_causal_conv1d_bwd(
    Tensor x, std::optional<Tensor> y, Tensor weight, Tensor dy,
    std::optional<Tensor> initial_state, std::optional<Tensor> dht,
    std::optional<Tensor> query_start_loc, int64_t activation,
    int64_t input_layout, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  const TensorMeta weight_meta = meta_of(weight);
  const char* layout =
      enum_name(kCausalConv1dBwdInputLayoutNames, input_layout);
  const std::vector<int64_t> qsl = int_values(query_start_loc);

  // TND/NTD carry one initial state per segment; the other layouts carry one
  // per batch row.
  const bool per_segment =
      std::strcmp(layout, "TND") == 0 || std::strcmp(layout, "NTD") == 0;
  const int64_t state_rows =
      per_segment ? (qsl.empty() ? 0 : static_cast<int64_t>(qsl.size()) - 1)
                  : size_of(x_meta, 0);

  Tensor out_dx = allocate_like(x_meta);
  Tensor out_dw = allocate_sizes(
      {size_of(weight_meta, 0), size_of(weight_meta, 1)},
      weight_meta.scalar_type, weight_meta);
  Tensor out_db = allocate_sizes({size_of(weight_meta, 1)},
                                 weight_meta.scalar_type, weight_meta);
  Tensor out_dinit = allocate_sizes(
      {state_rows, size_of(weight_meta, 0), size_of(weight_meta, 1)},
      x_meta.scalar_type, x_meta);

  FLA_STABLE_EXEC("aclnnCausalConv1dBwd", x_meta, stream, tensor(x_meta),
                  optional_tensor(y), tensor(weight_meta), tensor(meta_of(dy)),
                  optional_tensor(initial_state), optional_tensor(dht),
                  int_array(qsl), scalar(activation), CStrArg(layout),
                  out_tensor(meta_of(out_dx)), out_tensor(meta_of(out_dw)),
                  out_tensor(meta_of(out_db)), out_tensor(meta_of(out_dinit)));
  return std::make_tuple(out_dx, out_dw, out_db, out_dinit);
}

// ---------------------------------------------------------------------------
// npu_chunk_fwd_o
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_fwd_o =
    "npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor? g, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, float scale, int chunk_size, "
    "bool use_exp2, bool transpose_state_layout, int output_layout, "
    "int stream) -> Tensor";

Tensor run_npu_chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h,
                           std::optional<Tensor> g,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> chunk_indices, double scale,
                           int64_t chunk_size, bool use_exp2,
                           bool transpose_state_layout, int64_t output_layout,
                           int64_t stream) {
  const TensorMeta v_meta = meta_of(v);
  const char* layout = enum_name(kChunkFwdOOutputLayoutNames, output_layout);
  const int32_t dtype = v_meta.scalar_type;

  Tensor out;
  if (std::strcmp(layout, "BNSD") == 0) {
    out = allocate_sizes({size_of(v_meta, 0), size_of(v_meta, 1),
                          size_of(v_meta, 2), size_of(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "BSND") == 0) {
    out = allocate_sizes({size_of(v_meta, 0), size_of(v_meta, 2),
                          size_of(v_meta, 1), size_of(v_meta, 3)},
                         dtype, v_meta);
  } else if (std::strcmp(layout, "TND") == 0) {
    out = allocate_sizes({size_of(v_meta, 2), size_of(v_meta, 1),
                          size_of(v_meta, 3)},
                         dtype, v_meta);
  } else {
    out = allocate_sizes({size_of(v_meta, 1), size_of(v_meta, 2),
                          size_of(v_meta, 3)},
                         dtype, v_meta);
  }

  FLA_STABLE_EXEC("aclnnChunkFwdO", meta_of(v), stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(h)),
                  optional_tensor(g), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(scale), scalar(chunk_size),
                  scalar(use_exp2), scalar(transpose_state_layout),
                  CStrArg(layout), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_gdn_bwd_intra
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_gdn_bwd_intra =
    "npu_chunk_gdn_bwd_intra(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor A, Tensor d_o, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, float scale, int chunk_size, bool use_exp2, "
    "int stream) -> (Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor> run_npu_chunk_gdn_bwd_intra(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor A, Tensor d_o,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, bool use_exp2, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta v_meta = meta_of(v);
  Tensor out_dq = allocate_sizes(
      {size_of(q_meta, 0), size_of(v_meta, 1), size_of(q_meta, 2),
       size_of(q_meta, 3)},
      q_meta.scalar_type, q_meta);
  Tensor out_dk = allocate_like(v_meta);
  Tensor out_dv = allocate_like(v_meta);

  FLA_STABLE_EXEC("aclnnChunkGdnBwdIntra", q_meta, stream, tensor(q_meta),
                  tensor(meta_of(k)), tensor(v_meta), tensor(meta_of(g)),
                  tensor(meta_of(beta)), tensor(meta_of(A)),
                  tensor(meta_of(d_o)), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(scale), scalar(chunk_size),
                  scalar(use_exp2), out_tensor(meta_of(out_dq)),
                  out_tensor(meta_of(out_dk)), out_tensor(meta_of(out_dv)));
  return std::make_tuple(out_dq, out_dk, out_dv);
}

}  // namespace
