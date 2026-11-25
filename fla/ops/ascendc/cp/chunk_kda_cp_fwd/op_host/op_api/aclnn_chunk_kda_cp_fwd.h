// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#pragma once
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaCpFwdGetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* v, const aclTensor* gate, const aclTensor* beta,
    const aclTensor* aLog, const aclTensor* bias, const aclTensor* offsets, const char* group,
    int64_t ranks, int64_t rank, int64_t rounds, int64_t firstSource, bool sendTail, bool saveFinal, double scale,
    const aclTensor* output, const aclTensor* finalState, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnChunkKdaCpFwd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
