/*!
 * \file pre_process_fwd_kernel_merged.cpp
 * \brief pre_process_fwd_kernel_merged：薄入口：tiling 注册/解析 + workspace + AIC/AIV 分派
 *
 * 本文件由 kernel 主文件按"机械搬运"拆出（代码与拆分前逐字符相同）；
 * Stage/布局/同步协议的完整说明见 pre_process_fwd_kernel_merged_common.h 顶部注释与 docs/design.md。
 */

#include "pre_process_fwd_kernel_merged_common.h"
#include "pre_process_fwd_kernel_merged_vec.h"
#include "pre_process_fwd_kernel_merged_cube.h"
#ifndef TORCH_MODE
extern "C" __global__ __aicore__ void pre_process_fwd_kernel_merged(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR bg, GM_ADDR v,
    GM_ADDR cu_seqlens, GM_ADDR hm, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GDN::PreProcessFwdKernelMergedTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::PreProcessFwdKernelMergedTilingData, tilingData, tiling);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    (void)bg;   // DPLR（USE_BG）本轮不实现
    GDN::PpFwdCtx ctx;
    ctx.k = k;
    ctx.w = w;
    ctx.u = u;
    ctx.g = g;
    ctx.gk = gk;
    ctx.v = v;
    ctx.cu = cu_seqlens;
    ctx.hm = hm;
    ctx.ws = userWS;
    ctx.tiling = &tilingData;
    // TilingKey：1=USE_G / 2=USE_GK / 3=USE_BG（与 host 侧 SetTilingKey 一致）
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
            GDN::PpFwdCube cube(ctx);
            cube.Run();
        }
        if ASCEND_IS_AIV {
            GDN::PpFwdVector vec(ctx);
            vec.Run();
        }
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
            GDN::PpFwdCube cube(ctx);
            cube.Run();
        }
        if ASCEND_IS_AIV {
            GDN::PpFwdVector vec(ctx);
            vec.Run();
        }
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIC {
            GDN::PpFwdCube cube(ctx);
            cube.Run();
        }
        if ASCEND_IS_AIV {
            GDN::PpFwdVector vec(ctx);
            vec.Run();
        }
    }
}
#endif
