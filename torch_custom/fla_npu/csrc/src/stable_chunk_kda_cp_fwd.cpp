// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

namespace {
using torch::stable::Tensor;
constexpr const char* kSchema_chunk_kda_cp_fwd =
    "npu_chunk_kda_cp_fwd(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor a_log, Tensor bias, "
    "Tensor offsets, Tensor group_name, int rank_size, int rank, int rounds, int first_source, bool send_tail, "
    "bool save_final, float scale, int stream) -> (Tensor, Tensor)";

std::tuple<Tensor, Tensor> run_npu_chunk_kda_cp_fwd(
    Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor a_log, Tensor bias, Tensor offsets,
    Tensor group_name, int64_t rank_size, int64_t rank, int64_t rounds, int64_t first_source,
    bool send_tail, bool save_final, double scale, int64_t stream)
{
    namespace s = fla_npu_stable::stable;
    auto shape = s::meta_of(q);
    auto offsets_meta = s::meta_of(offsets);
    auto gate_meta = s::meta_of(gate);
    if (shape.ndim != 4 || shape.sizes[0] != 1 || shape.sizes[1] < 1 || shape.sizes[2] < 1 ||
        shape.sizes[3] != 128 || offsets_meta.ndim != 1 || offsets_meta.sizes[0] < 2) {
        throw std::invalid_argument("invalid fused KCP shape or local offsets shape");
    }
    const auto heads = shape.sizes[1];
    const auto tokens = shape.sizes[2];
    const auto segments = offsets_meta.sizes[0] - 1;
    auto output = s::allocate_sizes({1, tokens, heads, 128}, shape.scalar_type, shape);
    auto state = save_final ? s::allocate_sizes({segments, heads, 128, 128}, gate_meta.scalar_type, shape) :
                              s::allocate_sizes({1, 1, 1, 1}, gate_meta.scalar_type, shape);
    const std::string group = kcp_group_name(group_name);
    FLA_STABLE_EXEC("aclnnChunkKdaCpFwd", shape, stream,
                    s::nd_tensor(q), s::nd_tensor(k), s::nd_tensor(v), s::nd_tensor(gate), s::nd_tensor(beta),
                    s::nd_tensor(a_log), s::nd_tensor(bias), s::nd_tensor(offsets), s::CStrArg(group),
                    s::scalar(rank_size), s::scalar(rank), s::scalar(rounds), s::scalar(first_source),
                    s::scalar(send_tail), s::scalar(save_final), s::scalar(scale),
                    s::nd_tensor(output), s::nd_tensor(state));
    return {output, state};
}
} // namespace
