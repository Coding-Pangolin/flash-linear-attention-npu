// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#pragma once
#include "aclnn/aclnn_base.h"
extern "C" {
aclnnStatus aclnnKcpAllGatherMergeGetWorkspaceSize(
    const aclTensor *summary, const aclTensor *h, const char *group,
    int64_t rankSize, int64_t rank, const aclTensor *gathered,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnKcpAllGatherMerge(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
}
