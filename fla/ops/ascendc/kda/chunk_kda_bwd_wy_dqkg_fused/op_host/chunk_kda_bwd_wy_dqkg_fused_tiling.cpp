/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "chunk_kda_bwd_wy_dqkg_fused_tiling.h"

#include <algorithm>
#include <cstdlib>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
// Must match op_kernel/chunk_kda_bwd_wy_dqkg_fused_common.h SlotLayout*
constexpr int64_t MAX_BT = 64;
constexpr int64_t MAX_BK = 128; // match op_kernel USE_BK128=1
constexpr int64_t MAX_BV = 128; // match op_kernel USE_BV128=1
constexpr int64_t MAX_K_TOTAL = 128;
constexpr int64_t MAX_V_TOTAL = 256;
constexpr int64_t MAX_NBV = (MAX_V_TOTAL + MAX_BV - 1) / MAX_BV;
constexpr int64_t NUM_GM_SLOTS = 4;
constexpr int64_t WORKSPACE_ALIGN = 512;
constexpr int64_t MAX_HEAD = 128;

// SlotLayoutF32::TOTAL (element count)
constexpr int64_t SLOT_F32_TOTAL =
    /*dAWs*/ MAX_BT * MAX_BT +
    /*dASlot*/ MAX_NBV * MAX_BT * MAX_BT +
    /*dvbWs*/ MAX_BT * MAX_V_TOTAL +
    /*dqSlot*/ MAX_NBV * MAX_BT * MAX_BK +
    /*dkSlot*/ MAX_NBV * MAX_BT * MAX_BK +
    /*dwSlot*/ MAX_NBV * MAX_BT * MAX_BK +
    /*dADeltaWs*/ MAX_BT * MAX_BT +
    /*dkPartialWs*/ MAX_BT * MAX_BK +
    /*gkWs*/ MAX_BT * MAX_BK +
    /*dkgbWs*/ MAX_BT * MAX_BK +
    /*dgkWs*/ MAX_BK +
    /*dA3Ws*/ MAX_BT * MAX_BT +
    /*betaWs*/ MAX_BT +
    /*dbMergeWs*/ 2 * MAX_BT +
    /*dgkMergeWs*/ 2 * MAX_BK +
    /*dqGatedWs*/ MAX_BT * MAX_BK +
    /*kParkWs*/ MAX_BT * MAX_BK +
    /*gParkWs*/ MAX_BT * MAX_BK;

// SlotLayoutT::TOTAL — keep in sync with op_kernel common.h
constexpr int64_t SLOT_T_TOTAL =
    /*kgWs*/ MAX_BT * MAX_BK +
    /*dwNegWs*/ MAX_BT * MAX_BK +
    /*dAMaskedWs*/ MAX_BT * MAX_BT +
    /*dA2InterimWs*/ MAX_BT * MAX_BT +
    /*stateHWs*/ MAX_NBV * MAX_BV * MAX_BK +
    /*stateDhWs*/ MAX_NBV * MAX_BV * MAX_BK;

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_V = 2;
constexpr size_t INPUT_G = 4;
constexpr size_t INPUT_H = 7;
constexpr size_t INPUT_CU = 11;
constexpr size_t INPUT_IDX = 12;
constexpr size_t ATTR_SCALE = 0;
constexpr size_t ATTR_CHUNK = 1;
constexpr size_t ATTR_STATE_V_FIRST = 2;
} // namespace

ge::graphStatus Tiling4ChunkKdaBwdWyDqkgFused(gert::TilingContext *context)
{
    ChunkKdaBwdWyDqkgFusedTilingData tiling;
    auto qShapePtr = context->GetInputShape(INPUT_Q);
    auto vShapePtr = context->GetInputShape(INPUT_V);
    auto gShapePtr = context->GetInputShape(INPUT_G);
    auto hShapePtr = context->GetInputShape(INPUT_H);
    auto qDesc = context->GetInputDesc(INPUT_Q);
    if (qShapePtr == nullptr || vShapePtr == nullptr || gShapePtr == nullptr || hShapePtr == nullptr ||
        qDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &qShape = qShapePtr->GetStorageShape();
    const gert::Shape &vShape = vShapePtr->GetStorageShape();
    const gert::Shape &gShape = gShapePtr->GetStorageShape();
    const gert::Shape &hShape = hShapePtr->GetStorageShape();
    if (qShape.GetDimNum() != 4 || vShape.GetDimNum() != 4 || gShape.GetDimNum() != 4 || hShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = qShape.GetDim(0);
    const int64_t h = qShape.GetDim(1);
    const int64_t t = qShape.GetDim(2);
    const int64_t k = qShape.GetDim(3);
    const int64_t hv = vShape.GetDim(1);
    const int64_t v = vShape.GetDim(3);
    if (k != 128) {
        return ge::GRAPH_FAILED;
    }
    if (v != 128 && v != 256) {
        return ge::GRAPH_FAILED;
    }
    if (h <= 0 || hv <= 0 || hv < h || (hv % h) != 0 || h > MAX_HEAD || hv > MAX_HEAD) {
        return ge::GRAPH_FAILED;
    }
    if (vShape.GetDim(0) != batch || vShape.GetDim(2) != t) {
        return ge::GRAPH_FAILED;
    }
    if (gShape.GetDim(0) != batch || gShape.GetDim(1) != hv || gShape.GetDim(2) != t || gShape.GetDim(3) != k) {
        return ge::GRAPH_FAILED;
    }

    const float scale = *context->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE);
    const int64_t chunkSize = *context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK);
    const bool stateVFirst = *context->GetAttrs()->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    if (chunkSize != 64) {
        return ge::GRAPH_FAILED;
    }

    const auto cuShape = context->GetOptionalInputShape(INPUT_CU);
    const auto idxShape = context->GetOptionalInputShape(INPUT_IDX);
    const bool hasCu = cuShape != nullptr;
    const bool hasIdx = idxShape != nullptr;
    if (hasCu != hasIdx) {
        return ge::GRAPH_FAILED;
    }
    if (hasCu && batch != 1) {
        return ge::GRAPH_FAILED;
    }

    int64_t numChunks = 0;
    int64_t seqNum = batch;
    if (hasCu) {
        seqNum = cuShape->GetStorageShape().GetDim(0) - 1;
        const int64_t idxLen = idxShape->GetStorageShape().GetDim(0);
        if ((idxLen % 2) != 0 || idxLen <= 0) {
            return ge::GRAPH_FAILED;
        }
        numChunks = idxLen / 2;
    } else {
        numChunks = (t + chunkSize - 1) / chunkSize;
    }

    // h: [B,HV,NT,K,V] or [B,HV,NT,V,K]
    if (hShape.GetDim(0) != batch || hShape.GetDim(1) != hv || hShape.GetDim(2) != numChunks) {
        return ge::GRAPH_FAILED;
    }
    if (stateVFirst) {
        if (hShape.GetDim(3) != v || hShape.GetDim(4) != k) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (hShape.GetDim(3) != k || hShape.GetDim(4) != v) {
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t totalTasks = numChunks * batch;
    // F6: optional env override (no ACLNN ABI change). Production → Op Attr.
    //   FLA_WY_DQKG_STAGE=0|1|2|3  FLA_WY_DQKG_TASK_BEGIN=  FLA_WY_DQKG_TASK_END=
    int64_t stageId = 0;
    int64_t taskBegin = 0;
    int64_t taskEnd = totalTasks;
    if (const char *envStage = std::getenv("FLA_WY_DQKG_STAGE")) {
        stageId = static_cast<int64_t>(std::strtol(envStage, nullptr, 10));
    }
    if (const char *envBegin = std::getenv("FLA_WY_DQKG_TASK_BEGIN")) {
        taskBegin = static_cast<int64_t>(std::strtol(envBegin, nullptr, 10));
    }
    if (const char *envEnd = std::getenv("FLA_WY_DQKG_TASK_END")) {
        taskEnd = static_cast<int64_t>(std::strtol(envEnd, nullptr, 10));
    }
    if (stageId < 0 || stageId > 3) {
        stageId = 0;
    }
    if (taskBegin < 0) {
        taskBegin = 0;
    }
    if (taskEnd < 0 || taskEnd > totalTasks) {
        taskEnd = totalTasks;
    }
    if (taskBegin > taskEnd) {
        taskBegin = taskEnd;
    }
    const int64_t rangeTasks = std::max<int64_t>(taskEnd - taskBegin, (stageId == 0) ? totalTasks : 0);
    const int64_t schedTasks = (stageId == 0) ? totalTasks : rangeTasks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t blockDim =
        static_cast<uint32_t>(std::min<int64_t>(std::max<int64_t>(schedTasks, 1), coreNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    context->SetTilingKey(1);

    const uint64_t usedCore = static_cast<uint64_t>(blockDim == 0 ? 1 : blockDim);
    const int64_t elemBytes = (qDesc->GetDataType() == ge::DT_FLOAT16 || qDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;
    const int64_t slotBytes = SLOT_F32_TOTAL * static_cast<int64_t>(sizeof(float)) + SLOT_T_TOTAL * elemBytes;
    // Fused: 4 rolling banks. Stage A/B/C: one unique slot per head (=hv) so OpA can
    // finish all windows before OpB (no CrossCore across launches).
    const int64_t numSlots = (stageId == 0) ? NUM_GM_SLOTS : hv;
    int64_t wsBytes = static_cast<int64_t>(usedCore) * numSlots * slotBytes;
    wsBytes = (wsBytes + WORKSPACE_ALIGN - 1) / WORKSPACE_ALIGN * WORKSPACE_ALIGN;

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<uint64_t>(wsBytes);

    int64_t dataType = 0;
    if (qDesc->GetDataType() == ge::DT_BF16) {
        dataType = 1;
    }

    tiling.set_batch(batch);
    tiling.set_t(t);
    tiling.set_h(h);
    tiling.set_hv(hv);
    tiling.set_k(k);
    tiling.set_v(v);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_totalTasks(totalTasks);
    tiling.set_hasCuSeqlens(hasCu ? 1 : 0);
    tiling.set_hasChunkIndices(hasIdx ? 1 : 0);
    tiling.set_seqNum(seqNum);
    tiling.set_dataType(dataType);
    tiling.set_usedCoreNum(static_cast<int64_t>(usedCore));
    tiling.set_wsBytes(wsBytes);
    tiling.set_wsSlotBytes(slotBytes);
    tiling.set_stateVFirst(stateVFirst ? 1 : 0);
    tiling.set_bk(MAX_BK);
    tiling.set_bv(MAX_BV);
    tiling.set_scale(scale);
    tiling.set_stageId(stageId);
    tiling.set_taskBegin(stageId == 0 ? 0 : taskBegin);
    tiling.set_taskEnd(stageId == 0 ? totalTasks : taskEnd);
    tiling.set_numSlots(numSlots);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwdWyDqkgFused(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdWyDqkgFused)
    .Tiling(Tiling4ChunkKdaBwdWyDqkgFused)
    .TilingParse<ChunkKdaBwdWyDqkgFusedCompileInfo>(TilingPrepare4ChunkKdaBwdWyDqkgFused);

} // namespace optiling
