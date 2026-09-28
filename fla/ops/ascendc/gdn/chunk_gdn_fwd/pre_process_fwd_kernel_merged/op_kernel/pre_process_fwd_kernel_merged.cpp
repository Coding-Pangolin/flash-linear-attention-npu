/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged.cpp
 * \brief CP 前处理算子 kernel（v2：Cube/MIX 版，AIC 4 个 bf16 matmul + AIV elementwise）。
 *
 * 一个工作项 = 一条链（段 n × value-head hv），链内逐 chunk 串行。所有中间量放在
 * 本核 user workspace（GM），AIC/AIV 用 CrossCore flag 串联：
 *
 *   AIV  prologue : h←0（fp32+bf16）、m←I（fp32+bf16）        → kFlagState
 *   AIV  stage c  : 载入 W/k/left/v（bf16，尾块零填充）+ dg/decay → kFlagInputs
 *   AIC  ①③       : vTmp = W_c@bf16(h)、T1 = W_c@bf16(m)      → kFlagHalf1
 *   AIV           : v_new = (v - vTmp)·dg → bf16(v_new)；bf16(T1) → kFlagVNew
 *   AIC  ②        : dH = k_c^T @ bf16(v_new)                  → kFlagDH
 *   AIV           : h = decay⊙h + dH（同步存 bf16(h)）          → kFlagHUpd
 *   AIC  ④        : T2 = left^T @ bf16(T1)                    → kFlagT2
 *   AIV           : m = decay⊙m - T2（同步存 bf16(m)）          → kFlagState（下一 chunk）
 *
 * 语义与竞品/标杆对齐（见 docs/api.md、reference/reference.py）：
 *   * h 项用**未加门控的 k**，门控只作用在 v_new 上；m 项用 left = k·2^(g_last-g_t)；
 *   * h/m 的衰减：USE_G 为标量 2^(g_last)，USE_GK 为逐 k 的 2^(gk_last[k])；
 *   * m 链在 FP32 上做（这里的乘积把 m 量化到 bf16，与竞品 default/TF32 口径一致，
 *     在模型同构数据下实测 matched=1.000000）。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// CATLASS 的 arch 选择必须在包含 catlass 头之前给出（同 chunk_scaled_dot_kkt 的写法）
#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "pre_process_fwd_kernel_merged_struct.h"


namespace GDN {
using namespace AscendC;

constexpr int32_t CV_BT = 64;
constexpr int32_t CV_K = 128;
constexpr int32_t CV_V = 128;
constexpr int32_t CV_LANES = 8;

// ---------------- 每核 GM 暂存区（字节偏移）----------------
// [实验] 把 h 的 fp32 状态从偏移 0 挪到最后：CATLASS matmul 可能在本核 workspace 起始处
// 使用自己的暂存区，之前 hF32_（偏移 0 起）恒被冲成 0，而更靠后的 hBf_ 是对的。
constexpr int64_t WS_M_F32 = 0;                        // m  [K,K] fp32 65536
constexpr int64_t WS_H_BF = WS_M_F32 + 65536;          // bf16(h) [K,V] 32768
constexpr int64_t WS_M_BF = WS_H_BF + 32768;           // bf16(m) [K,K] 32768
constexpr int64_t WS_W_BF = WS_M_BF + 32768;           // W_c  [BT,K] bf16 16384
constexpr int64_t WS_K_BF = WS_W_BF + 16384;           // k_c  [BT,K] bf16 16384
constexpr int64_t WS_L_BF = WS_K_BF + 16384;           // left [BT,K] bf16 16384
constexpr int64_t WS_V_BF = WS_L_BF + 16384;           // v_c  [BT,V] bf16 16384
constexpr int64_t WS_VTMP_F32 = WS_V_BF + 16384;       // W@h   [BT,V] fp32 32768
constexpr int64_t WS_VNEW_BF = WS_VTMP_F32 + 32768;    // bf16(v_new) [BT,V] 16384
constexpr int64_t WS_DH_F32 = WS_VNEW_BF + 16384;      // dH [K,V] fp32 65536
constexpr int64_t WS_T1_F32 = WS_DH_F32 + 65536;       // T1 [BT,K] fp32 32768
constexpr int64_t WS_T1_BF = WS_T1_F32 + 32768;        // bf16(T1) [BT,K] 16384
constexpr int64_t WS_T2_F32 = WS_T1_BF + 16384;        // T2 [K,K] fp32 65536
// dH / T2 各双缓冲一份：AIC 写 chunk c 用的那份，AIV 在 chunk c 读 c-1 写的那份
constexpr int64_t WS_DH_F32_1 = WS_T2_F32 + 65536;     // dH 第二份 65536
constexpr int64_t WS_T2_F32_1 = WS_DH_F32_1 + 65536;   // T2 第二份 65536
constexpr int64_t WS_GATE = WS_T2_F32_1 + 65536;       // dg[BT] + decay[K] 1024
constexpr int64_t WS_H_F32 = WS_GATE + 4096;           // h  [K,V] fp32 65536（最后一段）
constexpr int64_t WS_GATE_DG = 0;
constexpr int64_t WS_GATE_DECAY = CV_BT * 4;

// ---------------- CrossCore flag（950 MIX 同核内 AIC <-> AIV）----------------
// ⚠ 必须用 mode 0x4（intra-block）+ **每个 AIV 子核各自的 flag slot**（第二个子核 =
//   id + 16），AIC 侧显式 wait/set 两个 slot。这与仓内 arch35 算子的约定一致
//   （见 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h、kda/chunk_kda_fwd/.../fwd_h.h）。
//   实测反例：用 A2/A3 风格的 `CrossCoreSetFlag<0x2, ...>` 时，950 上先干完的那个
//   子核就会把 AIC 放行，AIC 的 mm1/mm3 读到"只写了一半"的 h/m bf16 状态，
//   表现为 h 半边约一半行数据错、m 基本对（GDN 快路径暴露，KDA 慢路径看不出来）。
constexpr uint16_t PPFM_SUBFLAG_STRIDE = 16;   // AIV 子核 1 的 slot 偏移
constexpr uint16_t kFlagInputs = 1;   // AIV -> AIC：本 chunk staging 就位
constexpr uint16_t kFlagState = 2;    // AIV -> AIC：h/m 状态就位
constexpr uint16_t kFlagHalf1 = 3;    // AIC -> AIV：vTmp 与 T1 就位
constexpr uint16_t kFlagVNew = 4;     // AIV -> AIC：bf16(v_new) 与 bf16(T1) 就位
constexpr uint16_t kFlagDH = 5;       // AIC -> AIV：dH 与 T2 就位

// AIV 子核：写自己本地 slot（硬件按子核自动映射到 id / id+16）
__aicore__ inline void AivSetToAic(uint16_t id)
{
    // set_intra_block 不保证"之前的搬运已落地"，这里显式排空 MTE3
    PipeBarrier<PIPE_MTE3>();
    CrossCoreSetFlag<0x4, PIPE_MTE3>(id);
}

__aicore__ inline void AivWaitFromAic(uint16_t id)
{
    // 用 PIPE_S 排队：等待指令必须卡住后续指令的发射（否则后面的 MTE2/V 会先跑）
    CrossCoreWaitFlag<0x4, PIPE_S>(id);
}

// AIC：把两个 AIV 子核的 slot 都消费掉
__aicore__ inline void AicWaitFromAiv(uint16_t id)
{
    CrossCoreWaitFlag<0x4, PIPE_S>(id);
    CrossCoreWaitFlag<0x4, PIPE_S>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
}

// AIC：两个 slot 都要置位，否则只会唤醒一个子核
__aicore__ inline void AicSetToAiv(uint16_t id)
{
    PipeBarrier<PIPE_FIX>();
    CrossCoreSetFlag<0x4, PIPE_FIX>(id);
    CrossCoreSetFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
}

// ---------------- matmul 类型 ----------------
using MmAType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MmBType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MmCType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MmBiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulConfig CV_MM_CFG = GetNormalConfig(true);

// ---------------- CATLASS BlockMmad（950 的核内 cube→vector 惯用法）----------------
// 与仓内 chunk_scaled_dot_kkt 的 950 路径一致：BlockMmad + preSetFlags()/finalWaitFlags()
// 才是"C 已写回 GM"的保证；MatmulImpl::IterateAll 不提供这个保证（见 docs/validation.md §12.11）。
using MmArchTag = Catlass::Arch::Ascend950;
using MmDispatchPolicy = Catlass::Gemm::MmadPingpongTlaMulti<MmArchTag, true, false>;
using MmL1Shape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
using MmL0Shape = MmL1Shape;
// A 行主（matmul①③：W_c[BT,K] @ X[K,N]）
using MmTileCopyNT = Catlass::Gemm::Tile::PackedTileCopyTla<
    MmArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor, float,
    Catlass::layout::RowMajor>;
using MmBlockNT = Catlass::Gemm::Block::BlockMmadTla<MmDispatchPolicy, MmL1Shape, MmL0Shape, bfloat16_t,
                                                     bfloat16_t, float, void, MmTileCopyNT>;
// A 列主（matmul②④：left[BT,K]^T @ X[BT,N]）
using MmTileCopyTA = Catlass::Gemm::Tile::PackedTileCopyTla<
    MmArchTag, bfloat16_t, Catlass::layout::ColumnMajor, bfloat16_t, Catlass::layout::RowMajor, float,
    Catlass::layout::RowMajor>;
using MmBlockTA = Catlass::Gemm::Block::BlockMmadTla<MmDispatchPolicy, MmL1Shape, MmL0Shape, bfloat16_t,
                                                     bfloat16_t, float, void, MmTileCopyTA>;

// ---------------- AIV 侧 UB 布局（字节）----------------
// ⚠ 950 MIX 下 UB 由**一个 AIC + 两个 AIV 子核共享**（同 chunk_fwd_h / KDA fwd_h 的
//   架构约定）：任何"两个子核都会写"的 scratch 必须按 subIdx_ 切成两份，否则会出现
//   跨子核数据竞争（实测表现：h 状态被写坏、且随调度时快时慢 → 概率性错）。
//   只有"按段（off）分区"的缓冲（kBlk/wBlk/vBlk/scr）才可以被子核共享。
//   每个 per-subcore 常量都是**两份的总字节数**，取 Tensor 时按 subIdx_ 偏移一份。
constexpr int32_t PPFM_SUB = 2;      // AIV 子核数（UB 共享）
constexpr int32_t PPFM_SEG = 16;     // left / v_new 的段长（行）
constexpr int32_t PPFM_RB = 16;      // 状态更新的行块（行）

// ---------------- 诊断开关（定位概率性 h 错）----------------
// 打开后：每个工作项把前 N 个 chunk 的 "AIV 读到的 vTmpF_[0]"（AIV 侧）与
// "AIC 写出的 vTmpF_[0]"（AIC 侧）指纹写进 hm 的 m 半边第 0 行（覆盖该行，验收时排除）。
//   lane 0..3  = AIV 读到的 vTmpF_[0]（第 c 个 chunk）
//   lane 8..11 = AIC 写出的 vTmpF_[0]（第 c 个 chunk）
// 判读：两者不等 → AIV 读到别的代（跨核可见性/flag 提前）；相等但≠期望 → AIC 的 mm1 输入不对。
#ifndef PPFM_DIAG
#define PPFM_DIAG 1
#endif
constexpr int32_t PPFM_DIAG_CHUNKS = 4;
constexpr int64_t WS_DIAG = 626688;               // 每核 4 KiB（AIC 写，AIV epilogue 搬到 hm）

constexpr int32_t UB_ROW0_BF = 0;                                  // [K] bf16 ×2
constexpr int32_t UB_ROW1_BF = UB_ROW0_BF + PPFM_SUB * CV_K * 2;   // [K] bf16 ×2
constexpr int32_t UB_ROW2_BF = UB_ROW1_BF + PPFM_SUB * CV_K * 2;   // [K] bf16 ×2
constexpr int32_t UB_ROW0_F32 = UB_ROW2_BF + PPFM_SUB * CV_K * 2;  // [K] fp32 ×2
constexpr int32_t UB_ROW1_F32 = UB_ROW0_F32 + PPFM_SUB * CV_K * 4;
constexpr int32_t UB_ROW2_F32 = UB_ROW1_F32 + PPFM_SUB * CV_K * 4;
constexpr int32_t UB_DG = UB_ROW2_F32 + PPFM_SUB * CV_K * 4;       // [BT] fp32 ×2
constexpr int32_t UB_DECAY = UB_DG + PPFM_SUB * CV_BT * 4;         // [K] fp32 ×2
constexpr int32_t UB_EXP = UB_DECAY + PPFM_SUB * CV_K * 4;         // [8] fp32 ×2
constexpr int32_t UB_DECAY_PREV = UB_EXP + PPFM_SUB * CV_LANES * 4;  // [K] fp32 ×2
constexpr int32_t UB_GBLK = UB_DECAY_PREV + PPFM_SUB * CV_K * 4;   // [BT] fp32 ×2
constexpr int32_t UB_STATE_F = UB_GBLK + PPFM_SUB * CV_BT * 4;     // [RB,K] fp32 ×2
// extBlkF_：状态更新的 dH 暂存（RB 行）+ v_new 的 vTmp 暂存（2 段）→ 每子核取大者
constexpr int32_t UB_EXT_F = UB_STATE_F + PPFM_SUB * PPFM_RB * CV_V * 4;
constexpr int32_t UB_STATE_BF = UB_EXT_F + PPFM_SUB * 2 * PPFM_SEG * CV_V * 4;
// 下面这些按"段"分区，两个子核用不同 off，可共享
constexpr int32_t UB_KBLK_BF = UB_STATE_BF + PPFM_SUB * PPFM_RB * CV_V * 2;  // [BT,K] bf16
constexpr int32_t UB_WBLK_BF = UB_KBLK_BF + CV_BT * CV_K * 2;                // [BT,K] bf16
constexpr int32_t UB_VBLK_BF = UB_WBLK_BF + CV_BT * CV_K * 2;                // [BT,V] bf16
constexpr int32_t UB_SCR_F = UB_VBLK_BF + CV_BT * CV_V * 2;                  // [BT,K] fp32
constexpr int32_t UB_SCR_BF = UB_SCR_F + CV_BT * CV_K * 4;                   // [BT,K] bf16
constexpr int32_t UB_DBG = UB_SCR_BF + CV_BT * CV_K * 2;                     // 诊断槽 ×2
constexpr int32_t PPFM_VEC_UB_BYTES = UB_DBG + PPFM_SUB * 16 * 4;

// 上面都是**字节**偏移，取 Tensor 时要按元素大小换算（bf16 → /2，fp32 → /4）
constexpr int32_t UB_ROW0_BF_ELEM = UB_ROW0_BF / 2;
constexpr int32_t UB_ROW1_BF_ELEM = UB_ROW1_BF / 2;
constexpr int32_t UB_ROW2_BF_ELEM = UB_ROW2_BF / 2;
constexpr int32_t UB_ROW0_F_ELEM = UB_ROW0_F32 / 4;
constexpr int32_t UB_ROW1_F_ELEM = UB_ROW1_F32 / 4;
constexpr int32_t UB_ROW2_F_ELEM = UB_ROW2_F32 / 4;
constexpr int32_t UB_DG_ELEM = UB_DG / 4;
constexpr int32_t UB_DECAY_ELEM = UB_DECAY / 4;
constexpr int32_t UB_EXP_ELEM = UB_EXP / 4;
constexpr int32_t UB_DECAY_PREV_ELEM = UB_DECAY_PREV / 4;
constexpr int32_t UB_GBLK_ELEM = UB_GBLK / 4;
constexpr int32_t UB_STATE_F_ELEM = UB_STATE_F / 4;
constexpr int32_t UB_EXT_F_ELEM = UB_EXT_F / 4;
constexpr int32_t UB_STATE_BF_ELEM = UB_STATE_BF / 2;
constexpr int32_t UB_KBLK_BF_ELEM = UB_KBLK_BF / 2;
constexpr int32_t UB_WBLK_BF_ELEM = UB_WBLK_BF / 2;
constexpr int32_t UB_VBLK_BF_ELEM = UB_VBLK_BF / 2;
constexpr int32_t UB_SCR_F_ELEM = UB_SCR_F / 4;
constexpr int32_t UB_SCR_BF_ELEM = UB_SCR_BF / 2;

struct PpFwdCtx {
    GM_ADDR k = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR g = nullptr;
    GM_ADDR gk = nullptr;
    GM_ADDR v = nullptr;
    GM_ADDR cu = nullptr;
    GM_ADDR hm = nullptr;
    GM_ADDR ws = nullptr;
    const PreProcessFwdKernelMergedTilingData *tiling = nullptr;
};

// =====================================================================================
// AIV：elementwise
// =====================================================================================
class PpFwdVector {
public:
    __aicore__ inline PpFwdVector(const PpFwdCtx &ctx) : ctx_(ctx) {}

    __aicore__ inline void Run()
    {
        const auto *t = ctx_.tiling;
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.k));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.w));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            (ctx_.v == nullptr) ? ctx_.u : ctx_.v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.g));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.gk));
        cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ctx_.cu));
        hmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.hm));

        // AIV 子核在 MIX 下拿到的是 AIV 编号：换算回 AIC 编号（两个子核做同一份工作，
        // 但**必须切分工作**，否则两个子核互相覆盖；flag 计数天然平衡）。
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx()) / GetSubBlockNum();
        subIdx_ = static_cast<int32_t>(GetSubBlockIdx());
        subNum_ = static_cast<int32_t>(GetSubBlockNum());
        if (subNum_ <= 0) {
            subNum_ = 1;
        }
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        hF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_H_F32), CV_K * CV_V);
        mF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_M_F32), CV_K * CV_K);
        hBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_H_BF), CV_K * CV_V);
        mBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_M_BF), CV_K * CV_K);
        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        vBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_V_BF), CV_BT * CV_V);
        vTmpF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_VTMP_F32), CV_BT * CV_V);
        vNewBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_VNEW_BF), CV_BT * CV_V);
        dHF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32), CV_K * CV_V);
        dHF1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32_1), CV_K * CV_V);
        t1F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T1_F32), CV_BT * CV_K);
        t1Bf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_T1_BF), CV_BT * CV_K);
        t2F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32), CV_K * CV_K);
        t2F1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32_1), CV_K * CV_K);
        gateF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_GATE), CV_BT + CV_K);
#if PPFM_DIAG
        diagG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DIAG), 16);
#endif

        pipe_.InitBuffer(ubBuf_, PPFM_VEC_UB_BYTES);
        // per-subcore 视图：两个 AIV 子核共享同一块 UB，凡"两个子核都会写"的 scratch
        // 都按 subIdx_ 偏移一份，段分区缓冲（kBlk/wBlk/vBlk/scr）保持不偏移。
        const int32_t sbf = subIdx_ * CV_K;
        const int32_t sf = subIdx_ * CV_K;
        const int32_t sdg = subIdx_ * CV_BT;
        const int32_t sdecay = subIdx_ * CV_K;
        const int32_t sexp = subIdx_ * CV_LANES;
        const int32_t sstate = subIdx_ * PPFM_RB * CV_V;
        const int32_t sext = subIdx_ * 2 * PPFM_SEG * CV_V;
        row0Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW0_BF_ELEM + sbf];
        row1Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW1_BF_ELEM + sbf];
        row2Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW2_BF_ELEM + sbf];
        row0F_ = ubBuf_.Get<float>()[UB_ROW0_F_ELEM + sf];
        row1F_ = ubBuf_.Get<float>()[UB_ROW1_F_ELEM + sf];
        row2F_ = ubBuf_.Get<float>()[UB_ROW2_F_ELEM + sf];
        dgF_ = ubBuf_.Get<float>()[UB_DG_ELEM + sdg];
        decayF_ = ubBuf_.Get<float>()[UB_DECAY_ELEM + sdecay];
        expScratch_ = ubBuf_.Get<float>()[UB_EXP_ELEM + sexp];
        decayPrevF_ = ubBuf_.Get<float>()[UB_DECAY_PREV_ELEM + sdecay];
        gBlkF_ = ubBuf_.Get<float>()[UB_GBLK_ELEM + sdg];
        stateBlkF_ = ubBuf_.Get<float>()[UB_STATE_F_ELEM + sstate];
        extBlkF_ = ubBuf_.Get<float>()[UB_EXT_F_ELEM + sext];
        stateBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_STATE_BF_ELEM + sstate];
        kBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_KBLK_BF_ELEM];
        wBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_WBLK_BF_ELEM];
        vBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_VBLK_BF_ELEM];
        scrF_ = ubBuf_.Get<float>()[UB_SCR_F_ELEM];
        scrBf_ = ubBuf_.Get<bfloat16_t>()[UB_SCR_BF_ELEM];
#if PPFM_DIAG
        dbgF_ = ubBuf_.Get<float>()[UB_DBG / 4 + subIdx_ * 16];
#endif

        const int64_t taskNum = t->nSeq * t->Hv;
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {
            const int64_t hv = task % t->Hv;
            const int64_t n = task / t->Hv;
            const int64_t bos = cuGm_.GetValue(n);
            const int64_t eos = cuGm_.GetValue(n + 1);
            ProcessChain(n, hv, bos, eos - bos);
        }
    }

private:
    // ---- 调试用：把一段 GM 内容用向量 Cast 转 fp32 后取前 2 个值 ----
    __aicore__ inline void ProbeBf16(const GlobalTensor<bfloat16_t> &src, int32_t idx)
    {
        DataCopy(row2Bf_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        Cast(row1F_, row2Bf_, RoundMode::CAST_NONE, CV_K);
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(idx, row1F_.GetValue(0));
        row0F_.SetValue(idx + 1, row1F_.GetValue(1));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProbeF32(const GlobalTensor<float> &src, int32_t idx, int32_t lane)
    {
        DataCopy(row1F_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(idx, row1F_.GetValue(lane));
        row0F_.SetValue(idx + 1, row1F_.GetValue(lane + 1));
        PipeBarrier<PIPE_ALL>();
    }

    // 把 src[0] 用 DataCopy 回读到 UB 后存进 row1F_ 指定 lane（供 epilogue 带出）
    __aicore__ inline void ProbeState(const GlobalTensor<float> &src, int32_t dstLane)
    {
        DataCopy(row0F_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        row1F_.SetValue(dstLane, row0F_.GetValue(0));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float Exp2Scalar(float x)
    {
        expScratch_.SetValue(0, x * 0.6931471805599453f);
        PipeBarrier<PIPE_ALL>();
        Exp(expScratch_, expScratch_, CV_LANES);
        PipeBarrier<PIPE_ALL>();
        return expScratch_.GetValue(0);
    }

    __aicore__ inline void SetDecay(int64_t hv, int64_t tGlobal)
    {
        if (ctx_.tiling->gateMode == PPFM_GATE_USE_G) {
            const float glast = gGm_.GetValue(hv * ctx_.tiling->T + tGlobal);
            const float dc = Exp2Scalar(glast);
            for (int32_t k = 0; k < CV_K; ++k) {
                decayF_.SetValue(k, dc);
            }
        } else {
            for (int32_t k = 0; k < CV_K; ++k) {
                const float gk = gkGm_.GetValue((hv * ctx_.tiling->T + tGlobal) * CV_K + k);
                decayF_.SetValue(k, Exp2Scalar(gk));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessChain(int64_t n, int64_t hv, int64_t bos, int64_t len)
    {
        const auto *t = ctx_.tiling;
        // ---- prologue：h = 0，m = I ----
        Duplicate(row0F_, 0.0f, CV_V);
        PipeBarrier<PIPE_ALL>();
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            DataCopy(hF32_[r * CV_V], row0F_, CV_V);
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hBf_[r * CV_V], row0Bf_, CV_V);
            PipeBarrier<PIPE_ALL>();
        }
        // m 初值 = I：**逐行纯向量构造**（不再用共享的 identF_ 矩阵，避免两个子核
        // 在同一块 UB 上互相覆盖；也省下 64 KiB UB）。
        // ⚠ 不要用 row0F_.SetValue(r,1) 这类"标量写 UB + 向量写同一块 UB"的组合：
        //   实测标量写的落盘顺序不受 PipeBarrier<PIPE_V> 保护，会让个别行丢掉对角 1
        //   （表现为 m 只有 ~0.05% 元素错、max_abs≈1）。
        Duplicate(row2F_, 1.0f, CV_K);
        PipeBarrier<PIPE_V>();
        ArithProgression(row1F_, 0.0f, 1.0f, CV_K);   // row1F_[k] = k
        PipeBarrier<PIPE_V>();
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            Adds(row0F_, row1F_, -static_cast<float>(r), CV_K);   // k - r
            PipeBarrier<PIPE_V>();
            Abs(row0F_, row0F_, CV_K);
            PipeBarrier<PIPE_V>();
            Mins(row0F_, row0F_, 1.0f, CV_K);
            PipeBarrier<PIPE_V>();
            Sub(row0F_, row2F_, row0F_, CV_K);                    // 1 - min(|k-r|,1)
            PipeBarrier<PIPE_ALL>();
            DataCopy(mF32_[r * CV_K], row0F_, CV_K);
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, CV_K);
            PipeBarrier<PIPE_ALL>();
            DataCopy(mBf_[r * CV_K], row0Bf_, CV_K);
            PipeBarrier<PIPE_ALL>();
        }
        const int64_t nt = (len + CV_BT - 1) / CV_BT;
        for (int64_t c = 0; c < nt; ++c) {
            curChunk_ = c;
            const int64_t t0 = bos + c * CV_BT;
            const int64_t left = len - c * CV_BT;
            const int64_t rows = (left < CV_BT) ? left : CV_BT;
            if (c > 0) {
                // StageChunk 会覆盖 decayF_；先把上一 chunk 的 decay 存下来，
                // 供"推迟一个 chunk 的状态更新"使用
                for (int32_t k = 0; k < CV_K; ++k) {
                    decayPrevF_.SetValue(k, decayF_.GetValue(k));
                }
                PipeBarrier<PIPE_ALL>();
            }
            StageChunk(n, hv, t0, rows);
            if (c > 0) {
                // 上一 chunk 的 dH / T2 此刻早已落地（其间 AIC 已完成本 chunk 的 mm1/mm3）
                ApplyStateUpdates(true, c - 1);
            }
            AivSetToAic(kFlagState);
            UpdateVNew(hv, t0, rows);
            AivWaitFromAic(kFlagDH);
            // 读别的核（AIC）写的 GM 前必须让本核缓存行失效，否则会读到过期数据
            // （与 CANN matmul_client.h 中"读跨核 GM flag 前先 DCCI"的用法一致）
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF_);
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F_);
#ifdef PPFM_DEBUG_HEADER
            // [临时调试] 在各更新点之后立刻回读状态（fp32 标量直读，最可靠）
            // 结果先放 UB（row1F_ 高位 lane），最后由 epilogue 的 header 块带出
            ProbeState(hF32_, 100);
            ProbeState(mF32_, 101);
            ProbeState(dHF_, 102);
            ProbeState(t2F_, 103);
            PipeBarrier<PIPE_ALL>();
#endif
        }
        // 最后一个 chunk 的状态更新（此时 decayF_ 就是它的 decay，dH/T2 也已落地）
        ApplyStateUpdates(false, nt - 1);
        // ---- epilogue：写 hm ----
        const int64_t hmBase = ((n * t->Hv + hv) * CV_K) * (CV_V + CV_K);
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            DataCopy(row0F_, hF32_[r * CV_V], CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K)], row0F_, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(row0F_, mF32_[r * CV_K], CV_K);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V], row0F_, CV_K);
            PipeBarrier<PIPE_ALL>();
        }
#if PPFM_DIAG
        // 诊断收尾（只由子核 0 写）：lane 0..3 = AIV 读到的 vTmpF_[0]（第 c 个 chunk），
        // lane 8..11 = AIC 写出的 vTmpF_[0]。落在 hm 的 m 半边第 0 行（验收时排除该行）。
        if (subIdx_ == 0) {
            Duplicate(row0F_, 0.0f, CV_K);
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < PPFM_DIAG_CHUNKS; ++i) {
                row0F_.SetValue(i, dbgF_.GetValue(i));
            }
            PipeBarrier<PIPE_ALL>();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(diagG_);
            DataCopy(row1F_, diagG_, 8);          // 8 个 fp32 = 32B，满足对齐要求
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < PPFM_DIAG_CHUNKS; ++i) {
                row0F_.SetValue(8 + i, row1F_.GetValue(i));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + CV_V], row0F_, CV_K);
            PipeBarrier<PIPE_ALL>();
        }
#endif
#ifdef PPFM_DEBUG_HEADER
        // [临时调试] 收尾后覆盖 hm 第 0 行的 m 半边，写入 tiling/cu 关键值
        for (int32_t i = 0; i < CV_K; ++i) {
            row0F_.SetValue(i, 0.0f);
        }
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(0, static_cast<float>(t->nSeq));
        row0F_.SetValue(1, static_cast<float>(t->Hv));
        row0F_.SetValue(2, static_cast<float>(t->T));
        row0F_.SetValue(3, static_cast<float>(t->K));
        row0F_.SetValue(4, static_cast<float>(t->V));
        row0F_.SetValue(5, static_cast<float>(t->chunkSize));
        row0F_.SetValue(6, static_cast<float>(t->gateMode));
        row0F_.SetValue(7, static_cast<float>(t->usedAicNum));
        row0F_.SetValue(8, static_cast<float>(t->taskNum));
        row0F_.SetValue(9, static_cast<float>(bos));
        row0F_.SetValue(10, static_cast<float>(len));
        row0F_.SetValue(11, static_cast<float>(nt));
        row0F_.SetValue(12, static_cast<float>(GetBlockIdx()));
        row0F_.SetValue(13, static_cast<float>(GetSubBlockNum()));
        // 各阶段中间量抽样：bf16 一律走"向量 Cast 转 fp32"（标量 bf16 转换不可靠）
        ProbeBf16(kGm_, 20);            // 输入 k（第 0 行前 2 个）
        ProbeBf16(wGm_, 22);            // 输入 w
        ProbeBf16(vGm_, 24);            // 输入 v
        row0F_.SetValue(26, gGm_.GetValue(0));
        row0F_.SetValue(27, gGm_.GetValue(1));
        ProbeBf16(kBf_, 28);            // staging k
        ProbeBf16(wBf_, 30);            // staging w
        ProbeBf16(lBf_, 32);            // staging left
        ProbeBf16(vBf_, 34);            // staging v
        ProbeF32(gateF_, 36, 0);        // dg[0], dg[1]
        ProbeF32(gateF_, 38, CV_BT);    // decay[0], decay[1]
        ProbeF32(vTmpF_, 40, 0);        // matmul1 输出
        ProbeF32(t1F_, 42, 0);          // matmul3 输出
        ProbeF32(dHF_, 44, 0);          // matmul2 输出
        ProbeF32(t2F_, 46, 0);          // matmul4 输出
        ProbeF32(hF32_, 48, 0);         // h 状态
        ProbeF32(mF32_, 50, 0);         // m 状态
        ProbeBf16(vNewBf_, 52);         // bf16(v_new) 前 2 个
        ProbeBf16(t1Bf_, 54);           // bf16(T1) 前 2 个
        ProbeBf16(mBf_, 56);            // bf16(m) 第 0 行前 2 个
        ProbeBf16(hBf_, 58);            // bf16(h) 第 0 行前 2 个
        // 更新点回读（见 chunk 循环里的插桩，值放在 row1F_ 的 100..103 lane）
        row0F_.SetValue(52, row1F_.GetValue(100));   // hF32_[0] @ UpdateH 之后
        row0F_.SetValue(53, row1F_.GetValue(101));   // mF32_[0] @ UpdateM 之后
        row0F_.SetValue(54, row1F_.GetValue(102));   // dHF_[0] @ DH 之后
        row0F_.SetValue(55, row1F_.GetValue(103));   // t2F_[0] @ T2 之后
        PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(hmGm_[hmBase + CV_V], row0F_, CV_K);
        PipeBarrier<PIPE_ALL>();
#endif
    }

    // staging：W / k / left / v（bf16，尾块零填充）+ dg / decay
    __aicore__ inline void StageChunk(int64_t n, int64_t hv, int64_t t0, int64_t rows)
    {
        const auto *t = ctx_.tiling;
        const int64_t hk = hv / (t->hvPerHk == 0 ? 1 : t->hvPerHk);
        const bool useG = (t->gateMode == PPFM_GATE_USE_G);
        float glast = 0.0f;
        if (useG) {
            glast = gGm_.GetValue(hv * t->T + (t0 + rows - 1));
            // 向量化：dg[t] = exp2(glast - g[t])（整块一次算完，替代逐 token 标量 Exp）
            // ⚠ 尾块 rows 不是 8（32B）的整数倍时：DataCopy 的长度必须 32B 对齐，
            //    否则是 UB（越界读）；这里改用 DataCopyPad（blockLen 按字节给）。
            DataCopyExtParams gParams{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
            DataCopyPad(gBlkF_, gGm_[hv * t->T + t0], gParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Muls(gBlkF_, gBlkF_, -1.0f, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            Adds(gBlkF_, gBlkF_, glast, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            Muls(gBlkF_, gBlkF_, 0.6931471805599453f, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            // ⚠ 先整块清零再覆盖前 rows 个：Duplicate 的目的地址必须 32B 对齐，
            //    rows=36 时 dgF_[rows] 落在 144B（非 32B 整数倍）→ VEC 访问 UB 非对齐
            //    （error code 340）直接 aicore exception。
            Duplicate(dgF_, 0.0f, static_cast<int32_t>(CV_BT));
            PipeBarrier<PIPE_V>();
            Exp(dgF_, gBlkF_, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_ALL>();
        } else {
            // dgF_ 现在是 per-subcore 的私有 scratch，两个子核都要**各自填满**
            Duplicate(dgF_, 1.0f, static_cast<int32_t>(CV_BT));
        }
        PipeBarrier<PIPE_ALL>();
        SetDecay(hv, t0 + rows - 1);
        // ---- staging：**按整段（16 行）分配 subcore**，段内自己完成"清零/搬运/left 计算/落盘"----
        // 这样既没有两个 subcore 重复搬运，也不存在"块级算术读另一个 subcore 半成品"的竞态
        constexpr int32_t SEG = 16;
        for (int32_t seg = subIdx_; seg < CV_BT / SEG; seg += subNum_) {
            const int32_t off = seg * SEG;
            const int32_t valid = (rows > off) ? ((rows - off < SEG) ? (rows - off) : SEG) : 0;
            Duplicate(kBlkBf_[off * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
            Duplicate(wBlkBf_[off * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
            Duplicate(vBlkBf_[off * CV_V], static_cast<bfloat16_t>(0), SEG * CV_V);
            PipeBarrier<PIPE_V>();
            if (valid > 0) {
                DataCopy(kBlkBf_[off * CV_K], kGm_[(hk * t->T + t0 + off) * CV_K],
                         static_cast<uint32_t>(valid * CV_K));
                DataCopy(wBlkBf_[off * CV_K], wGm_[(hv * t->T + t0 + off) * CV_K],
                         static_cast<uint32_t>(valid * CV_K));
                DataCopy(vBlkBf_[off * CV_V], vGm_[(hv * t->T + t0 + off) * CV_V],
                         static_cast<uint32_t>(valid * CV_V));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(kBf_[off * CV_K], kBlkBf_[off * CV_K], SEG * CV_K);
            DataCopy(wBf_[off * CV_K], wBlkBf_[off * CV_K], SEG * CV_K);
            // 注意：v 不再落到 GM（v_new 直接从 UB 的 vBlkBf_ 读），省一份 16 KiB/chunk 的 MTE3
            PipeBarrier<PIPE_ALL>();
            // left：USE_G 为 bf16(k·dg)，USE_GK 为 k 本身
            if (useG) {
                Cast(scrF_[off * CV_K], kBlkBf_[off * CV_K], RoundMode::CAST_NONE, SEG * CV_K);
                PipeBarrier<PIPE_V>();
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(off + i) * CV_K], scrF_[(off + i) * CV_K], dgF_.GetValue(off + i), CV_K);
                }
                PipeBarrier<PIPE_V>();
                Cast(scrBf_[off * CV_K], scrF_[off * CV_K], RoundMode::CAST_RINT, SEG * CV_K);
                PipeBarrier<PIPE_ALL>();   // V -> MTE3
                DataCopy(lBf_[off * CV_K], scrBf_[off * CV_K], SEG * CV_K);
                PipeBarrier<PIPE_ALL>();
            } else {
                DataCopy(lBf_[off * CV_K], kBlkBf_[off * CV_K], SEG * CV_K);
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        // dg / decay 落 GM（供 kernel 自身调试与一致性检查）
        DataCopy(gateF_[WS_GATE_DG / 4], dgF_, CV_BT);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gateF_[WS_GATE_DECAY / 4], decayF_, CV_K);
        PipeBarrier<PIPE_ALL>();
        AivSetToAic(kFlagInputs);
    }

    __aicore__ inline void UpdateVNew(int64_t hv, int64_t t0, int64_t rows)
    {
        const bool useG = (ctx_.tiling->gateMode == PPFM_GATE_USE_G);
        AivWaitFromAic(kFlagHalf1);
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#if PPFM_DIAG
        // 诊断：记下"本子核读到的 vTmpF_[0]"（chunk 0 时它必须恰好是 0）
        if (curChunk_ < PPFM_DIAG_CHUNKS) {
            PipeBarrier<PIPE_ALL>();
            dbgF_.SetValue(static_cast<int32_t>(curChunk_), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
        // v_new = (v - vTmp) · dg → bf16（逐行；整块版本会引入 ~0.4% 的 GDN 偏差，待查）
        // v_new = (v - vTmp)·dg → bf16：同样**按整段分配 subcore**，段内做完 Cast/Sub/缩放/Cast/落盘
        constexpr int32_t SEG = 16;
        for (int32_t seg = subIdx_; seg < CV_BT / SEG; seg += subNum_) {
            const int32_t off = seg * SEG;
            // extBlkF_ 是 per-subcore 的（只有 2 段的空间），段内用本地偏移
            const int32_t lo = (seg - subIdx_) / subNum_ * SEG;
            DataCopy(extBlkF_[lo * CV_V], vTmpF_[off * CV_V], SEG * CV_V);
            PipeBarrier<PIPE_ALL>();
            Cast(scrF_[off * CV_V], vBlkBf_[off * CV_V], RoundMode::CAST_NONE, SEG * CV_V);
            PipeBarrier<PIPE_V>();
            Sub(scrF_[off * CV_V], scrF_[off * CV_V], extBlkF_[lo * CV_V], SEG * CV_V);
            PipeBarrier<PIPE_V>();
            if (useG) {
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(off + i) * CV_V], scrF_[(off + i) * CV_V], dgF_.GetValue(off + i), CV_V);
                }
                PipeBarrier<PIPE_V>();
            }
            Cast(scrBf_[off * CV_V], scrF_[off * CV_V], RoundMode::CAST_RINT, SEG * CV_V);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(vNewBf_[off * CV_V], scrBf_[off * CV_V], SEG * CV_V);
            PipeBarrier<PIPE_ALL>();
        }
        // bf16(T1)：同样按段分配
        for (int32_t seg = subIdx_; seg < CV_BT / SEG; seg += subNum_) {
            const int32_t off = seg * SEG;
            DataCopy(scrF_[off * CV_K], t1F_[off * CV_K], SEG * CV_K);
            PipeBarrier<PIPE_ALL>();
            Cast(scrBf_[off * CV_K], scrF_[off * CV_K], RoundMode::CAST_RINT, SEG * CV_K);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(t1Bf_[off * CV_K], scrBf_[off * CV_K], SEG * CV_K);
            PipeBarrier<PIPE_ALL>();
        }
        AivSetToAic(kFlagVNew);
    }

    // h = decay⊙h + dH ; m = decay⊙m - T2
    // usePrevDecay=true 时用上一 chunk 的 decay（配合"状态更新推迟一个 chunk"）
    __aicore__ inline void ApplyStateUpdates(bool usePrevDecay, int64_t dataChunk)
    {
        const bool useG = (ctx_.tiling->gateMode == PPFM_GATE_USE_G);
        // dH / T2 双缓冲：按数据所属 chunk 的奇偶选 buffer（不用拷贝，直接分支）
        const bool evenChunk = ((dataChunk & 1) == 0);
        GlobalTensor<float> &dhBuf = evenChunk ? dHF_ : dHF1_;
        GlobalTensor<float> &t2Buf = evenChunk ? t2F_ : t2F1_;
        // 每 RB 行一次搬运：块内逐行 Muls（廉价、无需栅栏），块级 Add/Sub/Cast
        constexpr int32_t RB = 16;
        for (int32_t rb = subIdx_ * RB; rb < CV_K; rb += subNum_ * RB) {
            DataCopy(stateBlkF_, hF32_[rb * CV_V], RB * CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(extBlkF_, dhBuf[rb * CV_V], RB * CV_V);
            PipeBarrier<PIPE_ALL>();
            for (int32_t r = rb; r < rb + RB; ++r) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(useG ? 0 : r)
                                              : (useG ? decayF_.GetValue(0) : decayF_.GetValue(r));
                Muls(stateBlkF_[(r - rb) * CV_V], stateBlkF_[(r - rb) * CV_V], dc, CV_V);
            }
            PipeBarrier<PIPE_V>();
            Add(stateBlkF_, stateBlkF_, extBlkF_, RB * CV_V);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(hF32_[rb * CV_V], stateBlkF_, RB * CV_V);
            PipeBarrier<PIPE_ALL>();
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * CV_V);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(hBf_[rb * CV_V], stateBlkBf_, RB * CV_V);
            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t rb = subIdx_ * RB; rb < CV_K; rb += subNum_ * RB) {
            DataCopy(stateBlkF_, mF32_[rb * CV_K], RB * CV_K);
            PipeBarrier<PIPE_ALL>();
            DataCopy(extBlkF_, t2Buf[rb * CV_K], RB * CV_K);
            PipeBarrier<PIPE_ALL>();
            for (int32_t r = rb; r < rb + RB; ++r) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(useG ? 0 : r)
                                              : (useG ? decayF_.GetValue(0) : decayF_.GetValue(r));
                Muls(stateBlkF_[(r - rb) * CV_K], stateBlkF_[(r - rb) * CV_K], dc, CV_K);
            }
            PipeBarrier<PIPE_V>();
            Sub(stateBlkF_, stateBlkF_, extBlkF_, RB * CV_K);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(mF32_[rb * CV_K], stateBlkF_, RB * CV_K);
            PipeBarrier<PIPE_ALL>();
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * CV_K);
            PipeBarrier<PIPE_ALL>();   // V -> MTE3
            DataCopy(mBf_[rb * CV_K], stateBlkBf_, RB * CV_K);
            PipeBarrier<PIPE_ALL>();
        }
    }

    const PpFwdCtx &ctx_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<bfloat16_t> row0Bf_;
    LocalTensor<bfloat16_t> row1Bf_;
    LocalTensor<bfloat16_t> row2Bf_;
    LocalTensor<float> row0F_;
    LocalTensor<float> row1F_;
    LocalTensor<float> row2F_;
    LocalTensor<float> dgF_;
    LocalTensor<float> decayF_;
    LocalTensor<float> expScratch_;
    LocalTensor<float> decayPrevF_;
    LocalTensor<float> gBlkF_;
    LocalTensor<float> stateBlkF_;
    LocalTensor<float> extBlkF_;
    LocalTensor<bfloat16_t> stateBlkBf_;
    LocalTensor<bfloat16_t> kBlkBf_;
    LocalTensor<bfloat16_t> wBlkBf_;
    LocalTensor<bfloat16_t> vBlkBf_;
    LocalTensor<float> scrF_;
    LocalTensor<bfloat16_t> scrBf_;
    int32_t subIdx_ = 0;
    int32_t subNum_ = 1;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> gkGm_;
    GlobalTensor<int64_t> cuGm_;
    GlobalTensor<float> hmGm_;
    GlobalTensor<float> hF32_;
    GlobalTensor<float> mF32_;
    GlobalTensor<bfloat16_t> hBf_;
    GlobalTensor<bfloat16_t> mBf_;
    GlobalTensor<bfloat16_t> wBf_;
    GlobalTensor<bfloat16_t> kBf_;
    GlobalTensor<bfloat16_t> lBf_;
    GlobalTensor<bfloat16_t> vBf_;
    GlobalTensor<float> vTmpF_;
    GlobalTensor<bfloat16_t> vNewBf_;
    GlobalTensor<float> dHF_;
    GlobalTensor<float> dHF1_;
    GlobalTensor<float> t1F_;
    GlobalTensor<bfloat16_t> t1Bf_;
    GlobalTensor<float> t2F_;
    GlobalTensor<float> t2F1_;
    GlobalTensor<float> gateF_;
#if PPFM_DIAG
    GlobalTensor<float> diagG_;      // 每核诊断区（读 AIC 写的 vTmp 指纹）
    LocalTensor<float> dbgF_;        // 每子核 16 个诊断槽
    int64_t curChunk_ = 0;
#endif
};

// =====================================================================================
// AIC：4 个 matmul
// =====================================================================================
class PpFwdCube {
public:
    __aicore__ inline PpFwdCube(const PpFwdCtx &ctx) : ctx_(ctx) {}

    __aicore__ inline void Run()
    {
        const auto *t = ctx_.tiling;
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        hBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_H_BF), CV_K * CV_V);
#if PPFM_DIAG
        diagG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DIAG), 16);
#endif
        mBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_M_BF), CV_K * CV_K);
        vNewBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_VNEW_BF), CV_BT * CV_V);
        t1Bf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_T1_BF), CV_BT * CV_K);
        vTmpF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_VTMP_F32), CV_BT * CV_V);
        dHF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32), CV_K * CV_V);
        dHF1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32_1), CV_K * CV_V);
        t1F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T1_F32), CV_BT * CV_K);
        t2F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32), CV_K * CV_K);
        t2F1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32_1), CV_K * CV_K);

        cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ctx_.cu));
        const int64_t taskNum = t->nSeq * t->Hv;
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {
            const int64_t hv = task % t->Hv;
            const int64_t n = task / t->Hv;
            const int64_t bos = cuGm_.GetValue(n);
            const int64_t eos = cuGm_.GetValue(n + 1);
            const int64_t len = eos - bos;
            const int64_t nt = (len + CV_BT - 1) / CV_BT;
            (void)hv;
            for (int64_t c = 0; c < nt; ++c) {
                ProcessChunk(c);
            }
        }
    }

private:
    __aicore__ inline void ProcessChunk(int64_t c)
    {
        // dH / T2 双缓冲：本 chunk 写到自己那一份（不用拷贝，直接分支）
        const bool evenChunk = ((c & 1) == 0);
        GlobalTensor<float> &dhBuf = evenChunk ? dHF_ : dHF1_;
        GlobalTensor<float> &t2Buf = evenChunk ? t2F_ : t2F1_;
        // ① vTmp[BT,V] = W_c[BT,K] @ bf16(h)[K,V]
        AicWaitFromAiv(kFlagInputs);
        AicWaitFromAiv(kFlagState);
        // ⚠ 读别的核（AIV）刚写过的 GM 之前必须让本核的 cache 失效：h/m 每 chunk 都被
        //    AIV 重写，若 AIC 命中自己缓存的旧行，mm1/mm3 就会拿到过期的 h/m
        //    （实测表现为概率性的 h 半边大面积错、m 只是略偏，且随调度时好时坏）。
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(wBf_);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(hBf_);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(mBf_);
        RunMmadNT(wBf_, hBf_, vTmpF_, CV_BT, CV_V, CV_K);

        // ③ T1[BT,K] = W_c[BT,K] @ bf16(m)[K,K]
        RunMmadNT(wBf_, mBf_, t1F_, CV_BT, CV_K, CV_K);
        AicSetToAiv(kFlagHalf1);

        // ② dH[K,V] = k_c^T @ bf16(v_new)[BT,V]
        AicWaitFromAiv(kFlagVNew);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(kBf_);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(lBf_);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vNewBf_);
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1Bf_);
        RunMmadTA(kBf_, vNewBf_, dhBuf, CV_K, CV_V, CV_BT);

        // ④ T2[K,K] = left^T @ bf16(T1)[BT,K]
        RunMmadTA(lBf_, t1Bf_, t2Buf, CV_K, CV_K, CV_BT);
        // dH 与 T2 都由 AIV 在**下一个 chunk 开头**使用，合并为一次跨核通知（省一次 flag 往返）
        AicSetToAiv(kFlagDH);
#if PPFM_DIAG
        if (c < PPFM_DIAG_CHUNKS) {
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(vTmpF_);
            PipeBarrier<PIPE_ALL>();
            diagG_.SetValue(static_cast<int32_t>(c), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
    }

    // A 行主：C[m,n] = A[m,k] @ B[k,n]（A/B 都是 bf16、行主；C 是 fp32 行主）
    __aicore__ inline void RunMmadNT(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                     GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> resource;
        MmBlockNT mm(resource);
        mm.preSetFlags();
        auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
        auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
        auto tA = tla::MakeTensor(gmA[0], layoutA, Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], layoutB, Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        mm(bA, bB, bC, shape);
        mm.finalWaitFlags();
    }

    // A 列主（= 逻辑 [m,k] 在内存里按 [k,m] 存，正好对应"转置 A"）：C[m,n] = A^T @ B
    __aicore__ inline void RunMmadTA(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                     GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> resource;
        MmBlockTA mm(resource);
        mm.preSetFlags();
        auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
        auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
        auto tA = tla::MakeTensor(gmA[0], layoutA, Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], layoutB, Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        mm(bA, bB, bC, shape);
        mm.finalWaitFlags();
    }

    const PpFwdCtx &ctx_;
    GlobalTensor<bfloat16_t> wBf_;
    GlobalTensor<bfloat16_t> kBf_;
    GlobalTensor<bfloat16_t> lBf_;
    GlobalTensor<bfloat16_t> hBf_;
    GlobalTensor<bfloat16_t> mBf_;
    GlobalTensor<bfloat16_t> vNewBf_;
    GlobalTensor<bfloat16_t> t1Bf_;
    GlobalTensor<float> vTmpF_;
    GlobalTensor<float> dHF_;
    GlobalTensor<float> dHF1_;
    GlobalTensor<float> t1F_;
    GlobalTensor<float> t2F_;
    GlobalTensor<float> t2F1_;
    GlobalTensor<int64_t> cuGm_;
#if PPFM_DIAG
    GlobalTensor<float> diagG_;      // 每核 4 KiB 诊断区（AIV epilogue 会搬到 hm）
#endif
};

} // namespace GDN

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
