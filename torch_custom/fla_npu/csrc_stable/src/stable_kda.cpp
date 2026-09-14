// Stable-ABI adapters: npu_kda_gate_cumsum, npu_chunk_kda_bwd_intra.
//
// These two cover the shapes a KDA-family operator runs into:
//
//   * kda_gate_cumsum -- optional tensors, an int_array, bool/double scalars,
//     and an output whose dtype differs from its source (fp32 over `g`).
//   * chunk_kda_bwd_intra -- ten tensor inputs, two int_arrays, a `char*`
//     enum argument carried as an int code plus a name table, and four outputs
//     that are allocated from their matching inputs.
//
// Included by stable_ops.cpp (single TU); registration lives there.

#include "thin_stable/at_facade.h"
#include "thin_stable/boxed.h"
#include "thin_stable/exec.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_thin::stable::TensorMeta;
using fla_npu_thin::stable::at_shim::kBFloat16;
using fla_npu_thin::stable::at_shim::kFloat;
using fla_npu_thin::stable::allocate_like;
using fla_npu_thin::stable::allocate_sizes;
using fla_npu_thin::stable::cstr;
using fla_npu_thin::stable::int_array;
using fla_npu_thin::stable::meta_of;
using fla_npu_thin::stable::optional_tensor;
using fla_npu_thin::stable::out_tensor;
using fla_npu_thin::stable::scalar;
using fla_npu_thin::stable::tensor;


// ---------------------------------------------------------------------------
// npu_kda_gate_cumsum
// ---------------------------------------------------------------------------

constexpr const char* kSchema_kda_gate_cumsum =
    "npu_kda_gate_cumsum(Tensor g, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, int chunk_size, bool use_gate_in_kernel, "
    "bool safe_gate, float lower_bound, int stream) -> Tensor";

Tensor run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,
                               std::optional<Tensor> dt_bias,
                               std::optional<Tensor> cu_seqlens,
                               int64_t chunk_size, bool use_gate_in_kernel,
                               bool safe_gate, double lower_bound,
                               int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  // The kernel accumulates in fp32 regardless of `g`'s dtype.
  Tensor out = allocate_sizes(g_meta.sizes, kFloat, g_meta);
  FLA_STABLE_EXEC("aclnnKdaGateCumsum", g_meta, stream, tensor(g_meta),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), scalar(chunk_size),
                  scalar(use_gate_in_kernel), scalar(safe_gate),
                  scalar(lower_bound), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_kda_bwd_intra
// ---------------------------------------------------------------------------

// The aclnn entry point takes `layout` as a string; the stable value
// conversions cannot carry one, so the caller passes a code and this table is
// the single source of the legal values.  The order must match the Python
// _char_code table -- tools/op_abi_parity.py checks exactly that.
constexpr const char* kChunkKdaBwdIntraLayoutNames[] = {"BSND", "BNSD"};

constexpr const char* kSchema_chunk_kda_bwd_intra =
    "npu_chunk_kda_bwd_intra(Tensor q, Tensor k, Tensor gk, Tensor beta, "
    "Tensor dAqk, Tensor dAkk, Tensor dq, Tensor dk, Tensor db, Tensor dg, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, bool safe_gate, "
    "int layout, int stream) -> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_chunk_kda_bwd_intra(
    Tensor q, Tensor k, Tensor gk, Tensor beta, Tensor dAqk, Tensor dAkk,
    Tensor dq, Tensor dk, Tensor db, Tensor dg,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, bool safe_gate, int64_t layout, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  // Each gradient output has the shape and dtype of its own input.
  Tensor out_dq = allocate_like(meta_of(dq));
  Tensor out_dk = allocate_like(meta_of(dk));
  Tensor out_db = allocate_like(meta_of(db));
  Tensor out_dg = allocate_like(meta_of(dg));
  FLA_STABLE_EXEC(
      "aclnnChunkKdaBwdIntra", q_meta, stream,
      tensor(meta_of(q)), tensor(meta_of(k)), tensor(meta_of(gk)),
      tensor(meta_of(beta)), tensor(meta_of(dAqk)), tensor(meta_of(dAkk)),
      tensor(meta_of(dq)), tensor(meta_of(dk)), tensor(meta_of(db)),
      tensor(meta_of(dg)), int_array(cu_seqlens), int_array(chunk_indices),
      scalar(chunk_size), scalar(safe_gate),
      cstr(kChunkKdaBwdIntraLayoutNames, layout), out_tensor(meta_of(out_dq)),
      out_tensor(meta_of(out_dk)), out_tensor(meta_of(out_db)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_db, out_dg);
}

}  // namespace
