/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details.
 */

#include "chunk_kda_fwd_intra_sub_chunk_tiling.h"

#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_G_IDX = 2;
constexpr size_t INPUT_CU_SEQLENS_IDX = 4;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 5;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr int64_t SUB_CHUNK_SIZE = 16;
constexpr int64_t MAX_K_DIM = 256;
constexpr int64_t MAX_HEAD = 128;
} // namespace

ge::graphStatus Tiling4ChunkKdaFwdIntraSubChunk(gert::TilingContext *context)
{
    ChunkKdaFwdIntraSubChunkTilingData tiling;
    auto qShapePtr = context->GetInputShape(INPUT_Q_IDX);
    auto gShapePtr = context->GetInputShape(INPUT_G_IDX);
    auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    if (qShapePtr == nullptr || gShapePtr == nullptr || qDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &qShape = qShapePtr->GetStorageShape();
    const gert::Shape &gShape = gShapePtr->GetStorageShape();
    if (qShape.GetDimNum() != 4 || gShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = qShape.GetDim(0);
    const int64_t h = qShape.GetDim(1);
    const int64_t t = qShape.GetDim(2);
    const int64_t k = qShape.GetDim(3);
    const int64_t hv = gShape.GetDim(1);
    if (k <= 0 || k > MAX_K_DIM || (k % 16) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (h <= 0 || hv <= 0 || hv < h || (hv % h) != 0 || h > MAX_HEAD || hv > MAX_HEAD) {
        return ge::GRAPH_FAILED;
    }
    if (gShape.GetDim(0) != batch || gShape.GetDim(2) != t || gShape.GetDim(3) != k) {
        return ge::GRAPH_FAILED;
    }

    const float scale = *context->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE_IDX);
    const int64_t chunkSize = *context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    if (chunkSize != 32 && chunkSize != 64 && chunkSize != 128) {
        return ge::GRAPH_FAILED;
    }
    if ((chunkSize % SUB_CHUNK_SIZE) != 0) {
        return ge::GRAPH_FAILED;
    }

    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto idxShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
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

    const int64_t numSubChunks = chunkSize / SUB_CHUNK_SIZE;
    const int64_t totalTasks = numChunks * numSubChunks * batch * hv;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(std::max<int64_t>(totalTasks, 1), coreNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    int64_t dataType = 0;
    if (qDesc->GetDataType() == ge::DT_FLOAT) {
        dataType = 2;
    } else if (qDesc->GetDataType() == ge::DT_BF16) {
        dataType = 1;
    }

    tiling.set_batch(batch);
    tiling.set_t(t);
    tiling.set_h(h);
    tiling.set_hv(hv);
    tiling.set_k(k);
    tiling.set_chunkSize(chunkSize);
    tiling.set_subChunkSize(SUB_CHUNK_SIZE);
    tiling.set_numChunks(numChunks);
    tiling.set_numSubChunks(numSubChunks);
    tiling.set_totalTasks(totalTasks);
    tiling.set_hasCuSeqlens(hasCu ? 1 : 0);
    tiling.set_hasChunkIndices(hasIdx ? 1 : 0);
    tiling.set_seqNum(seqNum);
    tiling.set_dataType(dataType);
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : static_cast<int64_t>(blockDim));
    tiling.set_scale(scale);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdIntraSubChunk(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdIntraSubChunk)
    .Tiling(Tiling4ChunkKdaFwdIntraSubChunk)
    .TilingParse<ChunkKdaFwdIntraSubChunkCompileInfo>(TilingPrepare4ChunkKdaFwdIntraSubChunk);

} // namespace optiling
