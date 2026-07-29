/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef OP_API_INC_ACLNN_CHUNK_KDA_BWD_WY_DQKG_FUSED_H
#define OP_API_INC_ACLNN_CHUNK_KDA_BWD_WY_DQKG_FUSED_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaBwdWyDqkgFusedGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *vNew, const aclTensor *g,
    const aclTensor *beta, const aclTensor *a, const aclTensor *h, const aclTensor *dh, const aclTensor *doGrad,
    const aclTensor *dv, const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, double scale,
    int64_t chunkSize, bool stateVFirst, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dv2Out,
    const aclTensor *dgOut, const aclTensor *dbOut, const aclTensor *dAOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaBwdWyDqkgFused(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
