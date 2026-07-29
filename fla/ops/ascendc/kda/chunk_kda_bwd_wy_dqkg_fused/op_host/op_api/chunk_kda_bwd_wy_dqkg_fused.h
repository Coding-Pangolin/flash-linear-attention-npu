/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef OP_API_INC_CHUNK_KDA_BWD_WY_DQKG_FUSED_H
#define OP_API_INC_CHUNK_KDA_BWD_WY_DQKG_FUSED_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 6> ChunkKdaBwdWyDqkgFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *vNew, const aclTensor *g,
    const aclTensor *beta, const aclTensor *a, const aclTensor *h, const aclTensor *dh, const aclTensor *doGrad,
    const aclTensor *dv, const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, float scale,
    int64_t chunkSize, bool stateVFirst, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dv2Out,
    const aclTensor *dgOut, const aclTensor *dbOut, const aclTensor *dAOut, aclOpExecutor *executor);
} // namespace l0op

#endif
