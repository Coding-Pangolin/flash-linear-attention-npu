/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "chunk_kda_bwd_wy_dqkg_fused.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdWyDqkgFused);

const std::array<const aclTensor *, 6> ChunkKdaBwdWyDqkgFused(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *vNew, const aclTensor *g,
    const aclTensor *beta, const aclTensor *a, const aclTensor *h, const aclTensor *dh, const aclTensor *doGrad,
    const aclTensor *dv, const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, float scale,
    int64_t chunkSize, bool stateVFirst, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dv2Out,
    const aclTensor *dgOut, const aclTensor *dbOut, const aclTensor *dAOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdWyDqkgFused, q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlensOptional,
           chunkIndicesOptional, scale, chunkSize, stateVFirst, dqOut, dkOut, dv2Out, dgOut, dbOut, dAOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdWyDqkgFused,
        OP_INPUT(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(dqOut, dkOut, dv2Out, dgOut, dbOut, dAOut), OP_ATTR(scale, chunkSize, stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdWyDqkgFused failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dv2Out, dgOut, dbOut, dAOut};
}
} // namespace l0op
