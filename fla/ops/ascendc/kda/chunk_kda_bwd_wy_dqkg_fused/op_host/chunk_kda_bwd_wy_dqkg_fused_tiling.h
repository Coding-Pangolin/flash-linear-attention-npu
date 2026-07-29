/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_TILING_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkKdaBwdWyDqkgFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hv);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, v);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, numChunks);
TILING_DATA_FIELD_DEF(int64_t, totalTasks);
TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
TILING_DATA_FIELD_DEF(int64_t, hasChunkIndices);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, dataType); // 0=fp16, 1=bf16
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, wsBytes);
TILING_DATA_FIELD_DEF(int64_t, wsSlotBytes);
TILING_DATA_FIELD_DEF(int64_t, stateVFirst);
TILING_DATA_FIELD_DEF(int64_t, bk);
TILING_DATA_FIELD_DEF(int64_t, bv);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaBwdWyDqkgFused, ChunkKdaBwdWyDqkgFusedTilingData)

struct ChunkKdaBwdWyDqkgFusedCompileInfo {};
} // namespace optiling

#endif
