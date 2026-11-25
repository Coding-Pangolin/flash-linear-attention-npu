// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include <string>
#include <tuple>

namespace {
using torch::stable::Tensor;
constexpr const char* kSchema_kcp_all_gather_merge =
    "npu_kcp_all_gather_merge(Tensor summary, Tensor(a!) h, Tensor group_name, "
    "int rank_size, int rank, int stream) -> (Tensor(a!), Tensor)";
std::string kcp_group_name(const Tensor& group_name)
{
    namespace s = fla_npu_stable::stable;
    const auto bytes = s::int_values(std::optional<Tensor>(group_name));
    std::string group;
    for (int64_t byte : bytes) {
        if (byte <= 0 || byte > 255) throw std::invalid_argument("invalid HCCL group name bytes");
        group.push_back(static_cast<char>(byte));
    }
    if (group.empty()) throw std::invalid_argument("empty HCCL group name");
    return group;
}

std::tuple<Tensor, Tensor> run_npu_kcp_all_gather_merge(
    Tensor summary, Tensor h, Tensor group_name, int64_t rank_size, int64_t rank, int64_t stream)
{
    namespace s = fla_npu_stable::stable;
    auto meta = s::meta_of(summary);
    if (meta.ndim != 3 || meta.sizes[0] < 1 || meta.sizes[0] > 16 ||
        meta.sizes[1] != 128 || meta.sizes[2] != 256 || rank_size < 2 || rank_size > 8 ||
        rank < 0 || rank >= rank_size) throw std::invalid_argument("invalid KCP metadata");
    const std::string group = kcp_group_name(group_name);
    auto gathered = s::allocate_sizes({rank_size, meta.sizes[0], 128, 256}, meta.scalar_type, meta);
    FLA_STABLE_EXEC("aclnnKcpAllGatherMerge", meta, stream, s::nd_tensor(summary),
                    s::nd_tensor(h), s::CStrArg(group), s::scalar(rank_size), s::scalar(rank),
                    s::nd_tensor(gathered));
    return {h, gathered};
}

} // namespace
