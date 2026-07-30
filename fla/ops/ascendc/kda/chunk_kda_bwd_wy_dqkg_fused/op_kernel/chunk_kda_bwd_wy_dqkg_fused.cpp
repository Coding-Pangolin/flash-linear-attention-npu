/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaBwdWyDqkgFused — BNSD + optional TND varlen, GVA.
 * tiling key 1: KERNEL_TYPE_MIX_AIC_1_2
 *
 * Math: Triton `chunk_kda_bwd_kernel_wy_dqkg_fused`
 *   (flash-linear-attention/fla/ops/kda/chunk_bwd.py L124-296). See
 *   chunk_kda_bwd_wy_dqkg_fused_common.h for the full stage-graph description.
 *
 * Inputs : q,k,v,v_new,g,beta,A(=Akk),h,dh,do,dv (+ cu_seqlens, chunk_indices)
 * Outputs: dq(fp32),dk(fp32),dv2,dg(fp32),db(fp32),dA(fp32)
 *
 * Dual-path: Ascend950 (__CCE_AICORE__==310) → arch35/ (regbase vector);
 *            910B / else → parent *_common/_cube/_vector.h.
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_bwd_wy_dqkg_fused_common.h"
#include "arch35/chunk_kda_bwd_wy_dqkg_fused_cube.h"
#include "arch35/chunk_kda_bwd_wy_dqkg_fused_vector.h"
#else
#include "chunk_kda_bwd_wy_dqkg_fused_common.h"
#include "chunk_kda_bwd_wy_dqkg_fused_cube.h"
#include "chunk_kda_bwd_wy_dqkg_fused_vector.h"
#endif

extern "C" __global__ __aicore__ void chunk_kda_bwd_wy_dqkg_fused(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR g, GM_ADDR beta, GM_ADDR a, GM_ADDR h, GM_ADDR dh,
    GM_ADDR doGrad, GM_ADDR dv, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv2,
    GM_ADDR dg, GM_ADDR db, GM_ADDR dA, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if (tilingData.dataType == 1) {
            if ASCEND_IS_AIC {
                kda_wy_dqkg::KdaWyDqkgCube<bfloat16_t> op;
                op.Init(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db, dA,
                       userWS, tilingData, &pipe);
                op.Process();
            }
            if ASCEND_IS_AIV {
                kda_wy_dqkg::KdaWyDqkgVector<bfloat16_t> op;
                op.Init(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db, dA,
                       userWS, tilingData, &pipe);
                op.Process();
            }
        } else {
            if ASCEND_IS_AIC {
                kda_wy_dqkg::KdaWyDqkgCube<half> op;
                op.Init(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db, dA,
                       userWS, tilingData, &pipe);
                op.Process();
            }
            if ASCEND_IS_AIV {
                kda_wy_dqkg::KdaWyDqkgVector<half> op;
                op.Init(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db, dA,
                       userWS, tilingData, &pipe);
                op.Process();
            }
        }
    }
}
