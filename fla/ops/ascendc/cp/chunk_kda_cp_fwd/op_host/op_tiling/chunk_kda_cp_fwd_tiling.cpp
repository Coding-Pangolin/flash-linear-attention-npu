// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include <algorithm>
#include <cmath>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "hccl/hccl_types.h"
#include "op_common/log/log.h"
#include "../../op_kernel/chunk_kda_cp_fwd_tiling.h"

namespace optiling {
static ge::graphStatus TilingKcpFwd(gert::TilingContext* context)
{
    const auto* qs = context->GetInputShape(0);
    const auto* offsets = context->GetInputShape(7);
    const auto* attrs = context->GetAttrs();
    auto* td = context->GetTilingData<ChunkKdaCpFwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, qs);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsets);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    const auto* group = attrs->GetAttrPointer<char>(0);
    const auto* ranks = attrs->GetAttrPointer<int64_t>(1);
    const auto* rank = attrs->GetAttrPointer<int64_t>(2);
    const auto* rounds = attrs->GetAttrPointer<int64_t>(3);
    const auto* src = attrs->GetAttrPointer<int64_t>(4);
    const auto* tail = attrs->GetAttrPointer<bool>(5);
    const auto* save = attrs->GetAttrPointer<bool>(6);
    const auto* scale = attrs->GetAttrPointer<float>(7);
    if (!group || !ranks || !rank || !rounds || !src || !tail || !save || !scale) {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = qs->GetStorageShape();
    const auto& offsetsShape = offsets->GetStorageShape();
    if (shape.GetDimNum() != 4 || offsetsShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t heads = shape.GetDim(1);
    const int64_t tokens = shape.GetDim(2);
    const int64_t segments = offsetsShape.GetDim(0) - 1;
    const bool relay = *rounds == KcpWorkspace::kStateRelayPlan;
    if (relay && (heads != 2 * KcpWorkspace::kRelayBatchHeads || segments != 1 || (*ranks != 4 && *ranks != 8) ||
        *src != (*rank == 0 ? -1 : 0) || *tail != (*rank + 1 < *ranks))) {
        return ge::GRAPH_FAILED;
    }
    if (shape.GetDim(0) != 1 || shape.GetDim(3) != 128 || heads <= 0 || heads > 128 ||
        tokens <= 0 || tokens > std::numeric_limits<int32_t>::max() || segments <= 0 || segments > tokens ||
        (*ranks != 2 && *ranks != 4 && *ranks != 8) || *rank < 0 || *rank >= *ranks ||
        *rounds < 0 || (*rounds > 2 && !relay) || (*rounds == 2 && heads < 2) ||
        *src < -1 || *src >= *rank || (*tail && *rank + 1 >= *ranks) ||
        (*rounds == 0 && (*tail || *src >= 0)) || !group[0] || !std::isfinite(*scale)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "metadata", "invalid shape or plan",
            "require BNSD(1,H,T,128), local offsets and a valid CP2/4/8 rank plan");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    constexpr uint32_t control = 1;
    const uint32_t cores = platform.GetCoreNumAic();
    if (platform.GetCurNpuArch() != NpuArch::DAV_3510 || cores <= control) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tasks = relay ? KcpWorkspace::kRelayBatchHeads : heads * segments;
    const uint32_t workers = static_cast<uint32_t>(std::min<int64_t>(cores - control, tasks));
    if (relay && workers < KcpWorkspace::kRelayBatchHeads) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t scratchOffset = *rounds == 0 || relay ? 0 : (*ranks + 1) * heads * KcpWorkspace::kSummaryBytes;
    // Relay uses one send batch and two receive batches with one slot per rank.
    const uint64_t scratch = relay ? (1 + 2 * (*ranks)) * KcpWorkspace::kRelayBatchBytes :
        scratchOffset + workers * KcpWorkspace::kWorkerBytes;
    if (scratch > 200000000) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace", std::to_string(scratch),
            "KCP worker and communication scratch exceeds 200000000 bytes");
        return ge::GRAPH_FAILED;
    }
    td->heads = heads;
    td->tokens = tokens;
    td->segments = segments;
    td->workers = workers;
    td->ranks = *ranks;
    td->rank = *rank;
    td->rounds = relay ? 2 : *rounds;
    td->stateRelay = relay;
    td->sendTail = *tail;
    td->firstSource = *src;
    td->saveFinal = *save;
    td->scale = *scale;
    td->workerOffset = scratchOffset;
    // MC2 executor creation consumes the resource metadata even when the
    // packed plan schedules zero data collectives in the device kernel.
    AscendC::Mc2CcTilingConfig cc(group, HCCL_CMD_ALLGATHER, "AllGather=level0:fullmesh",
        0, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, 6);
    if (cc.GetTiling(td->mc2InitTiling) || cc.GetTiling(td->mc2CcTiling)) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(workers + control);
    // Every vector block participates in the communication phase barriers.
    // Launch the full block set together when another stream is active.
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(10);
    auto* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = scratch + platform.GetLibApiWorkSpaceSize();
    context->GetRawTilingData()->SetDataSize(sizeof(ChunkKdaCpFwdTilingData));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(ChunkKdaCpFwd).Tiling(TilingKcpFwd);
} // namespace optiling
