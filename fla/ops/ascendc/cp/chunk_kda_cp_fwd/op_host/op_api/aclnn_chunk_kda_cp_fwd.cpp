// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "aclnn_chunk_kda_cp_fwd.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"
#include <initializer_list>
#include <string>

namespace {
bool Matches(const aclTensor* tensor, op::DataType dtype, std::initializer_list<int64_t> dims)
{
    const auto& shape = tensor->GetViewShape();
    if (tensor->GetDataType() != dtype || shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}
} // namespace

extern "C" {
aclnnStatus aclnnInnerChunkKdaCpFwdGetWorkspaceSize(
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclTensor*, const aclTensor*, const char*,
    int64_t, int64_t, int64_t, int64_t, bool, bool, double, const aclTensor*, const aclTensor*,
    uint64_t*, aclOpExecutor**);
aclnnStatus aclnnInnerChunkKdaCpFwd(void*, uint64_t, aclOpExecutor*, aclrtStream);
void __attribute__((weak)) NnopbaseSetHcclServerType(void*, uint32_t);

aclnnStatus aclnnChunkKdaCpFwdGetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* v, const aclTensor* gate, const aclTensor* beta,
    const aclTensor* aLog, const aclTensor* bias, const aclTensor* offsets, const char* group,
    int64_t ranks, int64_t rank, int64_t rounds, int64_t firstSource, bool sendTail, bool saveFinal, double scale,
    const aclTensor* output, const aclTensor* finalState, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    for (const auto* tensor : {q, k, v, gate, beta, aLog, bias, offsets, output, finalState}) {
        if (!tensor) {
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (!op::IsContiguous(tensor) || op::IsPrivateFormat(tensor->GetStorageFormat()) ||
            op::IsPrivateFormat(tensor->GetViewFormat())) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!group || !group[0] || !workspaceSize || !executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto& qShape = q->GetViewShape();
    const auto& offsetsShape = offsets->GetViewShape();
    if (qShape.GetDimNum() != 4 || offsetsShape.GetDimNum() != 1 || offsetsShape.GetDim(0) < 2) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t h = qShape.GetDim(1), t = qShape.GetDim(2);
    const int64_t segments = offsetsShape.GetDim(0) - 1;
    const bool finalMatch = saveFinal ? Matches(finalState, op::DataType::DT_FLOAT, {segments, h, 128, 128}) :
                                       Matches(finalState, op::DataType::DT_FLOAT, {1, 1, 1, 1});
    if (!Matches(q, op::DataType::DT_BF16, {1, h, t, 128}) ||
        !Matches(k, op::DataType::DT_BF16, {1, h, t, 128}) ||
        !Matches(v, op::DataType::DT_BF16, {1, h, t, 128}) ||
        !Matches(gate, op::DataType::DT_FLOAT, {1, h, t, 128}) ||
        !Matches(beta, op::DataType::DT_FLOAT, {1, h, t}) ||
        !Matches(aLog, op::DataType::DT_FLOAT, {h}) ||
        !Matches(bias, op::DataType::DT_FLOAT, {h, 128}) ||
        offsets->GetDataType() != op::DataType::DT_INT32 ||
        !Matches(output, op::DataType::DT_BF16, {1, t, h, 128}) || !finalMatch) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnChunkKdaCpFwd", "tensors", "invalid shape or dtype",
            "expected BNSD raw inputs and outputs matching the local packed plan");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return aclnnInnerChunkKdaCpFwdGetWorkspaceSize(q, k, v, gate, beta, aLog, bias, offsets, group, ranks, rank,
        rounds, firstSource, sendTail, saveFinal, scale, output, finalState, workspaceSize, executor);
}

aclnnStatus aclnnChunkKdaCpFwd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // nnopbase's CCU service enum differs from the device Hccl enum.
    constexpr uint32_t kNnopbaseCcu = 2;
    if (!NnopbaseSetHcclServerType) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnChunkKdaCpFwd", "HCCL service", "unavailable",
            "CANN must export NnopbaseSetHcclServerType for CCU");
        return ACLNN_ERR_INNER;
    }
    NnopbaseSetHcclServerType(executor, kNnopbaseCcu);
    return aclnnInnerChunkKdaCpFwd(workspace, workspaceSize, executor, stream);
}
} // extern "C"
