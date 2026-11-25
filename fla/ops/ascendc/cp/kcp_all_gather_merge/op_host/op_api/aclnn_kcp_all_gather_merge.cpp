// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "aclnn_kcp_all_gather_merge.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_common/log/log.h"
#include <initializer_list>
#include <string>
#include <utility>

namespace {
// Executor-side nnopbase service IDs. The device Hccl template uses a
// separate enum, where HCCL_SERVER_TYPE_CCU is 5.
constexpr uint32_t kNnopbaseAicpu = 0;
constexpr uint32_t kNnopbaseCcu = 2;

std::string KcpShapeString(const op::Shape &shape)
{
    std::string text = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i) text += ",";
        text += std::to_string(shape.GetDim(i));
    }
    return text + "]";
}
} // namespace

extern "C" {
aclnnStatus aclnnInnerKcpAllGatherMergeGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const char *, int64_t, int64_t,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
aclnnStatus aclnnInnerKcpAllGatherMerge(void *, uint64_t, aclOpExecutor *, aclrtStream);
void __attribute__((weak)) NnopbaseSetHcclServerType(void *, uint32_t);

aclnnStatus aclnnKcpAllGatherMergeGetWorkspaceSize(
    const aclTensor *summary, const aclTensor *h, const char *group,
    int64_t rankSize, int64_t rank, const aclTensor *gathered,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    constexpr const char *name = "aclnnKcpAllGatherMerge";
    const char *missing = !summary ? "summary" : !h ? "h" : !group ? "group" :
        !gathered ? "gathered" : !workspaceSize ? "workspaceSize" : !executor ? "executor" : nullptr;
    if (missing) {
        OP_LOGE_WITH_INVALID_INPUT(name, missing);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto &s = summary->GetViewShape();
    const auto &hs = h->GetViewShape();
    const auto &gs = gathered->GetViewShape();
    if (!group[0]) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, "group", "empty", "require an initialized HCCL group name");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (rankSize < 2 || rankSize > 8 || rank < 0 || rank >= rankSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, "rankSize/rank",
            std::to_string(rankSize) + "/" + std::to_string(rank), "require P=2..8 and 0<=rank<P");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (s.GetDimNum() != 3 || s.GetDim(0) < 1 || s.GetDim(0) > 16 || s.GetDim(1) != 128 || s.GetDim(2) != 256) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, "summary.shape", KcpShapeString(s), "require [H,128,256], H=1..16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hs.GetDimNum() != 3 || hs.GetDim(0) != s.GetDim(0) || hs.GetDim(1) != 128 || hs.GetDim(2) != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, "h.shape", KcpShapeString(hs), "require [H,128,128], H matches summary");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gs.GetDimNum() != 4 || gs.GetDim(0) != rankSize || gs.GetDim(1) != s.GetDim(0) ||
        gs.GetDim(2) != 128 || gs.GetDim(3) != 256) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, "gathered.shape", KcpShapeString(gs), "require [P,H,128,256]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto &entry : {std::make_pair(summary, "summary"), std::make_pair(h, "h"), std::make_pair(gathered, "gathered")}) {
        const auto *t = entry.first;
        if (t->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, entry.second, std::to_string(static_cast<int>(t->GetDataType())), "dtype must be FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!op::IsContiguous(t) || op::IsPrivateFormat(t->GetStorageFormat()) || op::IsPrivateFormat(t->GetViewFormat())) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(name, entry.second,
                "contiguous=" + std::to_string(op::IsContiguous(t)) + ",format=" + std::to_string(static_cast<int>(t->GetStorageFormat())),
                "require contiguous public-format storage and view");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return aclnnInnerKcpAllGatherMergeGetWorkspaceSize(summary, h, group, rankSize, rank,
                                                     h, gathered, workspaceSize, executor);
}

aclnnStatus aclnnKcpAllGatherMerge(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    const uint32_t service = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510
        ? kNnopbaseCcu : kNnopbaseAicpu;
    if (!NnopbaseSetHcclServerType) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnKcpAllGatherMerge", "HCCL service", "unavailable",
            "CANN must export NnopbaseSetHcclServerType for the selected MC2 service");
        return ACLNN_ERR_INNER;
    }
    NnopbaseSetHcclServerType(executor, service);
    return aclnnInnerKcpAllGatherMerge(workspace, workspaceSize, executor, stream);
}
} // extern C
