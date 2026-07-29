/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "aclnn_chunk_kda_bwd_wy_dqkg_fused.h"
#include "chunk_kda_bwd_wy_dqkg_fused.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MAX_HEAD = 128;

struct Params {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *vNew = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *a = nullptr;
    const aclTensor *h = nullptr;
    const aclTensor *dh = nullptr;
    const aclTensor *doGrad = nullptr;
    const aclTensor *dv = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool stateVFirst = false;
    const aclTensor *dqOut = nullptr;
    const aclTensor *dkOut = nullptr;
    const aclTensor *dv2Out = nullptr;
    const aclTensor *dgOut = nullptr;
    const aclTensor *dbOut = nullptr;
    const aclTensor *dAOut = nullptr;
};

aclnnStatus ContiguousInPlace(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < Rank(lhs); ++idx) {
        if (Dim(lhs, idx) != Dim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckParams(const Params &p)
{
    CHECK_COND(p.q && p.k && p.v && p.vNew && p.g && p.beta && p.a && p.h && p.dh && p.doGrad && p.dv,
               ACLNN_ERR_PARAM_INVALID, "all required inputs must be non-null.");
    CHECK_COND(p.dqOut && p.dkOut && p.dv2Out && p.dgOut && p.dbOut && p.dAOut, ACLNN_ERR_PARAM_INVALID,
               "all outputs must be non-null.");
    CHECK_COND(p.chunkSize == 64, ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64.");
    CHECK_COND(Rank(p.q) == 4 && Rank(p.k) == 4 && Rank(p.v) == 4 && Rank(p.g) == 4 && Rank(p.beta) == 3 &&
                   Rank(p.a) == 4 && Rank(p.h) == 5,
               ACLNN_ERR_PARAM_INVALID, "BNSD ranks: q/k/v/g/a rank4, beta rank3, h rank5.");
    CHECK_COND(SameShape(p.q, p.k), ACLNN_ERR_PARAM_INVALID, "q/k shape mismatch.");

    const int64_t B = Dim(p.q, 0);
    const int64_t H = Dim(p.q, 1);
    const int64_t T = Dim(p.q, 2);
    const int64_t K = Dim(p.q, 3);
    const int64_t HV = Dim(p.v, 1);
    const int64_t V = Dim(p.v, 3);
    CHECK_COND(K == 128, ACLNN_ERR_PARAM_INVALID, "K must be 128.");
    CHECK_COND(V == 128 || V == 256, ACLNN_ERR_PARAM_INVALID, "V must be 128 or 256.");
    CHECK_COND(H > 0 && HV >= H && (HV % H) == 0 && H <= MAX_HEAD && HV <= MAX_HEAD, ACLNN_ERR_PARAM_INVALID,
               "invalid H/HV.");
    CHECK_COND(SameShape(p.v, p.vNew) && SameShape(p.v, p.doGrad) && SameShape(p.v, p.dv), ACLNN_ERR_PARAM_INVALID,
               "v/v_new/dox/dv must share shape.");
    CHECK_COND(Dim(p.g, 0) == B && Dim(p.g, 1) == HV && Dim(p.g, 2) == T && Dim(p.g, 3) == K, ACLNN_ERR_PARAM_INVALID,
               "g must be [B,HV,T,K].");
    CHECK_COND(Dim(p.beta, 0) == B && Dim(p.beta, 1) == HV && Dim(p.beta, 2) == T, ACLNN_ERR_PARAM_INVALID,
               "beta must be [B,HV,T].");
    CHECK_COND(Dim(p.a, 0) == B && Dim(p.a, 1) == HV && Dim(p.a, 2) == T && Dim(p.a, 3) == p.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "a must be [B,HV,T,chunkSize].");
    CHECK_COND(Dim(p.dqOut, 0) == B && Dim(p.dqOut, 1) == HV && Dim(p.dqOut, 2) == T && Dim(p.dqOut, 3) == K &&
                   p.dqOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "dqOut must be [B,HV,T,K] float32.");
    CHECK_COND(SameShape(p.dqOut, p.dkOut) && SameShape(p.dqOut, p.dgOut), ACLNN_ERR_PARAM_INVALID,
               "dq/dk/dg outs must match.");
    CHECK_COND(SameShape(p.dv2Out, p.v) && p.dv2Out->GetDataType() == p.v->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "dv2Out must match v.");
    CHECK_COND(Dim(p.dbOut, 0) == B && Dim(p.dbOut, 1) == HV && Dim(p.dbOut, 2) == T &&
                   p.dbOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "dbOut must be [B,HV,T] float32.");
    CHECK_COND(Dim(p.dAOut, 0) == B && Dim(p.dAOut, 1) == HV && Dim(p.dAOut, 2) == T &&
                   Dim(p.dAOut, 3) == p.chunkSize && p.dAOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "dAOut must be [B,HV,T,BT] float32.");

    const bool hasCu = p.cuSeqlensOptional != nullptr;
    const bool hasIdx = p.chunkIndicesOptional != nullptr;
    CHECK_COND(hasCu == hasIdx, ACLNN_ERR_PARAM_INVALID, "cu_seqlens and chunk_indices must be paired.");
    if (hasCu) {
        CHECK_COND(B == 1, ACLNN_ERR_PARAM_INVALID, "varlen requires B=1.");
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkKdaBwdWyDqkgFusedGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *vNew, const aclTensor *g,
    const aclTensor *beta, const aclTensor *a, const aclTensor *h, const aclTensor *dh, const aclTensor *doGrad,
    const aclTensor *dv, const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, double scale,
    int64_t chunkSize, bool stateVFirst, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dv2Out,
    const aclTensor *dgOut, const aclTensor *dbOut, const aclTensor *dAOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // Include scalar attrs in DFX_IN: when K==V, h/dh shapes are identical for
    // stateVFirst true/false, so omitting stateVFirst lets L2 reuse the wrong tiling.
    L2_DFX_PHASE_1(aclnnChunkKdaBwdWyDqkgFused,
                   DFX_IN(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlensOptional, chunkIndicesOptional,
                          scale, chunkSize, stateVFirst),
                   DFX_OUT(dqOut, dkOut, dv2Out, dgOut, dbOut, dAOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    Params params{q,     k,     v,     vNew,  g,     beta,  a,      h,      dh,     doGrad, dv,
                  cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, stateVFirst,
                  dqOut, dkOut, dv2Out, dgOut, dbOut, dAOut};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(ContiguousInPlace(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.vNew, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dh, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.doGrad, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dqOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dkOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dv2Out, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dgOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dbOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.dAOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::ChunkKdaBwdWyDqkgFused(
        params.q, params.k, params.v, params.vNew, params.g, params.beta, params.a, params.h, params.dh,
        params.doGrad, params.dv, params.cuSeqlensOptional, params.chunkIndicesOptional,
        static_cast<float>(params.scale), params.chunkSize, params.stateVFirst, params.dqOut, params.dkOut,
        params.dv2Out, params.dgOut, params.dbOut, params.dAOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaBwdWyDqkgFused(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdWyDqkgFused);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
