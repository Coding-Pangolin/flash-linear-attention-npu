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
// A5 的 L0C→UB 直连（手写 tile 级 mmad 用；A2/A3 用 PackedTileCopyTla 落 GM）
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "pre_process_fwd_kernel_merged_struct.h"


namespace GDN {
using namespace AscendC;

// ---------------- 目标 arch 分档 ----------------
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define PPFM_ARCH_IS_950 1
// 950：有 L0C→UB 直连通道（A2 优化方案可用）
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 1
#endif
#else
#define PPFM_ARCH_IS_950 0
// 910B/910_93(A2/A3)：cube↔vector 必须经 GM，无 L0C→UB 通道
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 0
#endif
#endif
// 跨核 flag 模式：950 用 0x4（同 block 内 AIC↔AIV，每子核 slot），910B 用 0x2
#if PPFM_ARCH_IS_950
constexpr int32_t PPFM_XCORE_MODE = 0x4;
#else
constexpr int32_t PPFM_XCORE_MODE = 0x2;
#endif

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
// ITER7（P3 跨 chunk 预取）：kBf_/lBf_ 的第二槽（chunk 奇偶选槽）
constexpr int64_t WS_K_BF_1 = WS_H_F32 + 65536;        // k_c 第二槽 16384
constexpr int64_t WS_L_BF_1 = WS_K_BF_1 + 16384;       // left 第二槽 16384
constexpr int64_t WS_GATE_DG = 0;
constexpr int64_t WS_GATE_DECAY = CV_BT * 4;

// ---------------- CrossCore flag（MIX 内 AIC <-> 2×AIV）----------------
// 两个平台的同步模型不同（见 PPFM_XCORE_MODE / PPFM_ARCH_IS_950）：
//   * 950 用 mode 0x4：flag ID 按 subblock 分槽（第二个子核 = id + 16），AIC 侧显式
//     wait/set 两个 slot。与仓内 arch35 算子的约定一致
//     （见 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h、kda/chunk_kda_fwd/.../fwd_h.h）。
//     实测反例：用 A2/A3 风格的 `CrossCoreSetFlag<0x2, ...>` 时，950 上先干完的那个
//     子核就会把 AIC 放行，AIC 的 mm1/mm3 读到"只写了一半"的 h/m bf16 状态，
//     表现为 h 半边约一半行数据错、m 基本对（GDN 快路径暴露，KDA 慢路径看不出来）。
//   * 910B/910_93 用 mode 0x2：AIC 与「本 block 的 2 个 AIV」是集合同步 —— AIC 的一次
//     set 对本 block 两个 AIV 同时置起；两个 AIV 都 set 同一个 ID 才算 AIC 侧事件置起。
//     因此 AIC 侧每轮只需一对 set/wait（多 set/多 wait 会让 flag 计数失衡）。
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
    // 对称于 AIC 侧：本核写出的 GM（kBf_/wBf_/lBf_/vNewBf_/t1Bf_/状态）也要先对其他核可见
#if PPFM_LEGACY_CACHEOPS
    DataSyncBarrier<MemDsbT::DDR>();
#endif
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_MTE3>(id);
}

__aicore__ inline void AivWaitFromAic(uint16_t id)
{
    // 950：用 PIPE_S 排队 —— 等待指令卡住后续指令的发射（否则后面的 MTE2/V 会先跑）。
    // 910B/910_93：消费方是 MTE2（从 GM 回读 cube 的 C），按 A2 惯用法在 MTE2 上排队
    //   （见仓内 arch22 的 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(..._READY_FLAG)`）。
    //   实测 A2 上用 PIPE_S 等待时，随后的 MTE2 会提前发射，读到"半新半旧"的 C：
    //   表现为 h 半边误差 ~1.5e-2 且逐次小幅跳动（v=0 的探针本应得到 h≡0）。
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
#else
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}

// AIC：消费 AIV 侧的通知
//   950 (mode 0x4)：flag ID 按 subblock 分槽，两个 AIV 子核各自 set 自己的 ID ⇒ 这里要等两次
//   910B/910_93 (mode 0x2)：AIC 与「本 block 的 2 个 AIV」是**集合同步**——两个 AIV 必须
//     都 set 同一个 ID，事件才会对 AIC 置起 ⇒ 这里只等一次
//   （约定出处：仓内 chunk_fwd_h/op_kernel/chunk_fwd_h_policy.h 的 FwdHAicPeerFlag 注释）
__aicore__ inline void AicWaitFromAiv(uint16_t id)
{
#if PPFM_ARCH_IS_950
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_S>(id);
    CrossCoreWaitFlag<0x4, PIPE_S>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#else
    // 910B：AIC 侧消费方同样是 MTE2（把 AIV 写好的 k/w/h/m/vNew 搬进 L1/L0）
    CrossCoreWaitFlag<PPFM_XCORE_MODE, PIPE_MTE2>(id);
#endif
}

// AIC -> AIV：
//   950：两个 slot 都要置位，否则只会唤醒一个子核；
//   910B/910_93：0x2 下一次 set 即对本 block 的两个 AIV 同时置起，只能 set 一次
//   （多 set 会让计数失衡 —— 单条 flag 连续 set 超过 15 次会挂死，见 catlass cross_core_sync.hpp）
__aicore__ inline void AicSetToAiv(uint16_t id)
{
    PipeBarrier<PIPE_FIX>();
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(id);
#if PPFM_ARCH_IS_950
    CrossCoreSetFlag<PPFM_XCORE_MODE, PIPE_FIX>(static_cast<uint16_t>(id + PPFM_SUBFLAG_STRIDE));
#endif
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
#if PPFM_ARCH_IS_950
using MmArchTag = Catlass::Arch::Ascend950;
#else
using MmArchTag = Catlass::Arch::AtlasA2;
#endif
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

// ---------------- A5 手写 tile 级（L0C→UB）类型 ----------------
// 与仓内 chunk_fwd_h_cube.h / chunk_kda_fwd_fwd_h.h 同一套 API：
//   PackedTileCopyTlaToUB 提供 CopyGmToL1A/B、CopyL1ToL0A/B、CopyL0CToDst（落 UB）；
//   TileMmadTla 做单 tile MMAD。A2/A3 没有 UB 直连通道，仍用上面的 PackedTileCopyTla 落 GM。
#if PPFM_ARCH_IS_950
using TiledArchTag = Catlass::Arch::Ascend950;
#else
using TiledArchTag = Catlass::Arch::AtlasA2;
#endif
#if PPFM_ARCH_IS_950
using TiledCopyNT = Common::Tile::PackedTileCopyTlaToUB<
    TiledArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
using TiledCopyTA = Common::Tile::PackedTileCopyTlaToUB<
    TiledArchTag, bfloat16_t, Catlass::layout::ColumnMajor, bfloat16_t, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
#else
// 910B：没有 L0C→UB，这里只用它的 L1A 布局标签喂给 TileMmadTla
using TiledCopyNT = Catlass::Gemm::Tile::PackedTileCopyTla<
    TiledArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
using TiledCopyTA = Catlass::Gemm::Tile::PackedTileCopyTla<
    TiledArchTag, bfloat16_t, Catlass::layout::ColumnMajor, bfloat16_t, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor>;
#endif
#if PPFM_ARCH_IS_950
// ITER8（A2）：NT 形态的 C 直接落 UB，SPLIT_M（前一半行→低半区、后一半行→高半区）
using TiledCopyNTSplitUb = Common::Tile::PackedTileCopyTlaToUB<
    TiledArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor,
    float, Catlass::layout::RowMajor, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
#endif
using TiledMmadNT = Catlass::Gemm::Tile::TileMmadTla<TiledArchTag, bfloat16_t,
                                                     typename TiledCopyNT::LayoutTagL1A>;
using TiledMmadTA = Catlass::Gemm::Tile::TileMmadTla<TiledArchTag, bfloat16_t,
                                                     typename TiledCopyTA::LayoutTagL1A>;
// GM 落点的 tile+C 回写类型（A2/A3 与 A5 的 A1 增量都用它）
// C 落 GM 的 tile-copy 别名（两代 API 名字不同）
#if PPFM_ARCH_IS_950
template <class TensorC>
using MmCopyL0CToGm = typename MmTileCopyNT::template CopyL0CToDst<TensorC>;
template <class TensorC>
using MMTACopyL0CToGm = typename MmTileCopyTA::template CopyL0CToDst<TensorC>;
#else
template <class TensorC>
using MmCopyL0CToGm = typename MmTileCopyNT::template CopyL0CToGm<TensorC>;
template <class TensorC>
using MMTACopyL0CToGm = typename MmTileCopyTA::template CopyL0CToGm<TensorC>;
#endif
using MmTileMmadNT = Catlass::Gemm::Tile::TileMmadTla<MmArchTag, bfloat16_t,
                                                      typename MmTileCopyNT::LayoutTagL1A>;
using MmTileMmadTA = Catlass::Gemm::Tile::TileMmadTla<MmArchTag, bfloat16_t,
                                                      typename MmTileCopyTA::LayoutTagL1A>;
// 手写 tile 路径下 L1 的两个槽（A 在前、B 在后），单位字节
constexpr int32_t TILED_L1_A_OFF = 0;
constexpr int32_t TILED_L1_B_OFF = 32 * 1024;
// L1A/L1B 的「容量形状」：必须与 BlockMmad 的 L1_TILE_M/K/N 一致（zZ/nZ 分形布局的
// stride 由 originShape 决定，用实际 (m,k) 构造会让 GM→L1 的落点与 L1→L0 的读点错位，
// 表现为 mmad 读到空 L0、C 恒为 0）。TILED_L1_B_OFF=32KiB 正是 128x128 bf16 的 footprint。
constexpr int32_t TILED_L1_CAP_M = 128;
constexpr int32_t TILED_L1_CAP_K = 128;
constexpr int32_t TILED_L1_CAP_N = 128;
// 手写 tile 级 mmad 开关：1=用 TileMmadTla 手拼，0=退回 BlockMmadTla
// ✅ A1 数值已对齐（2026-09-28，241 device6 实测）：
//   - 只 tile mm1（SEL=1）时 m 半边与基线逐位一致，只 tile mm3（SEL=2）时 h 半边逐位一致
//     ⇒ 两个 matmul 各自的 tile 结果与 BlockMmad 等价；
//   - 全 tile 时 5 轮 smoke 有 3 轮命中**已知的 GDN h 跨核可见性窗口**（TILE=0 基线 5/5 干净），
//     误差幅度 1.17~1.53 随机跳动，属时序放大，待 A2（A5 L0C→UB 直连）结构性消除。
//   根因（曾表现为 h 半边错、m≈decay·I）：`CopyL0CToGmTla` 的 4 参调用会误选
//   `(dst, src, l0Batch, dstNdStride)` 批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，
//   C 恒为 workspace 初值 0。必须走 3 参 `(dst, src, unitFlag)`。
#ifndef PPFM_TILE_MMAD
#define PPFM_TILE_MMAD 1
#endif
// ITER9：A2 优化方案（mm1 的 C 由 fixpipe SPLIT_M 直落 UB）开关。
//   0 = A5 主线（C 落 GM，已验证）；1 = UB 落点（首测 h 半边崩，UB 语义待实测确认）。
#ifndef PPFM_VTMP_UB
#define PPFM_VTMP_UB 1
#endif
// UB 语义诊断（默认 0）：1=UB 落点同时再写一份 GM，并在 AIV 侧回采探针
#ifndef PPFM_VTMP_UB_DIAG
#define PPFM_VTMP_UB_DIAG 0
#endif
// 实验开关：1=保留手工 DCCI/DSB（历史做法）；0=只用跨核 flag（与生产算子一致）
// 950：实测只用跨核 flag 就够（并且去掉 DCCI 后竞态由 3/6 降到 1/6），默认 0。
// 910B/910_93：实测 AIC 会读到 AIV 尚未对其他核可见的 bf16(h)（chunk0 的 h≡0 探针
//   仍得到 1.6e-2 的 h），故先按 A2 老做法启用 DCCI/DSB；若后续定位到更精确的边，
//   可只保留必要的那一条（见 docs/a2_opt_status.md 的定位方法）。
#ifndef PPFM_LEGACY_CACHEOPS
#if PPFM_ARCH_IS_950
#define PPFM_LEGACY_CACHEOPS 0
#else
#define PPFM_LEGACY_CACHEOPS 1
#endif
#endif
// 临时诊断开关：1=在 prologue 给 AIC 要写的 C 缓冲预置哨兵（见 ProcessChain）
#ifndef PPFM_SENTINEL_PROBE
#define PPFM_SENTINEL_PROBE 0
#endif
// 临时诊断开关：1=把 AIV 读到的 vTmp 第 0 行搬到 hm 的 m 半边第 0 行（chain0/head0）
#ifndef PPFM_RD_PROBE
#define PPFM_RD_PROBE 0
#endif
// KDA 的逐 k 衰减（decay[k] = 2^gk_last[k]）是否走向量化实现。
// 0 = 原来的逐点 SetValue + Exp2Scalar（每 chunk 128 次"标量写 + 2 次全栅栏 + Exp + 标量读"）
// 1 = 整块 DataCopy gk_last → ×ln2 → Exp（与逐点版本逐位等价，L1 位级门禁验证）
#ifndef PPFM_KDA_DECAY_VEC
#define PPFM_KDA_DECAY_VEC 1
#endif
// T1（mm3 的 C）是否由 fixpipe 直接按 bf16 落 GM（=1）——省掉 AIV 侧"读回 t1F_(fp32) →
// Cast → 写 t1Bf_" 的整条回路（每 chunk 32KB 读 + 16KB 写 + 一次 32K 元素的 Cast）。
// 前提：fixpipe 的 fp32→bf16 量化与原来的 CAST_RINT 等价（L1 位级门禁验证）；
// 仅在 PPFM_TILE_MMAD=1（手写 tile 路径）下生效。
#ifndef PPFM_T1_FIXPIPE_BF16
#define PPFM_T1_FIXPIPE_BF16 1
#endif
#if !PPFM_TILE_MMAD
#undef PPFM_T1_FIXPIPE_BF16
#define PPFM_T1_FIXPIPE_BF16 0
#endif
// 满 chunk（rows == BT）时 AIC 直接读输入张量里的 w/k，省掉 AIV 每 chunk 的
// w 载入(16KiB)+w 落盘(16KiB)+k 落盘(16KiB) 与对应事件对；尾块仍走 staging
// 零填充路径。两侧都由 rows 判定，天然一致。
// ⚠ 实测**净负收益**（2026-09-29，两平台一致），故默认 0：
//   950  T=1024 211.6→217.7 µs(+2.9%)、T=4096 624.8→643.5 µs(+3.0%)、模型 case 3825→3891 µs(+1.7%)
//   910B T=4096 651.3→661.7 µs(+1.6%)、模型 case 4055→4164 µs(+2.7%)
//   猜测原因：AIV 的 staging 写在读侧把数据"预热"进了 L2（AIC 随后读的是热行），
//   改成读输入张量后 AIC 每次拿的是冷行；省下的 48 KiB MTE3 抵不过这次延迟变差。
#ifndef PPFM_AIC_DIRECT_INPUTS
#define PPFM_AIC_DIRECT_INPUTS 0
#endif
// 早期怀疑"C 的跨核可见性"时加的 4 处"过渡探读"（各读 8 个 fp32 并配一次 PIPE_ALL）。
// 现在可见性结论已明确（见 validation §11/§13：hBf_ 写坏、T1 双写等），这些探读疑似纯开销
// （每 chunk 最多 4 次全栅栏 + 4 次小 DataCopy）。默认 **1 = 保留**（不改主线行为）；
// 计划 R9 一轮把它们置 0 做 A/B，并用 L3 ≥20 独立进程确认不会让 §1.4 的残余竞态变差。
#ifndef PPFM_LEGACY_PROBE_READS
#define PPFM_LEGACY_PROBE_READS 1
#endif
#if !PPFM_TILE_MMAD
#undef PPFM_AIC_DIRECT_INPUTS
#define PPFM_AIC_DIRECT_INPUTS 0
#endif

// ---------------- AIV 侧跨流水同步：事件对（P1a）----------------
// 热路径原来用 PipeBarrier<PIPE_ALL> 把所有流水排空；跨流水的依赖其实只需要"生产者→消费者"
// 的事件对。PPFM_AIV_EVENTS=0 时退化成与原来等价的 PIPE_ALL（用于 A/B 与快速回退）。
// 事件 ID 分工（每个 SET 都有同 ID 的 WAIT，成对消耗；AIC 侧用的是它自己的一套，互不影响）：
//   ID0 MTE2->V   ID1 V->MTE3   ID2 MTE3->MTE2   ID3 MTE3->V
//   ID4 V->MTE2   ID5 MTE2->MTE3   ID6 V->S      ID7 S->V
// ⚠ 经验（见 docs/pipeline_parallel_plan.md §P1a）：**必须用事件对**，
// 用 PipeBarrier<PIPE_X> 代替 PIPE_ALL 会丢跨流水依赖（ITER6 曾 6/6 全错）。
#ifndef PPFM_AIV_EVENTS
#define PPFM_AIV_EVENTS 1
#endif
#if PPFM_AIV_EVENTS
#define AIV_SET_MTE2_V()     SetFlag<HardEvent::MTE2_V>(EVENT_ID0)
#define AIV_WAIT_MTE2_V()    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0)
#define AIV_SET_V_MTE3()     SetFlag<HardEvent::V_MTE3>(EVENT_ID1)
#define AIV_WAIT_V_MTE3()    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1)
#define AIV_SET_MTE3_MTE2()  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2)
#define AIV_WAIT_MTE3_MTE2() WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2)
#define AIV_SET_MTE3_V()     SetFlag<HardEvent::MTE3_V>(EVENT_ID3)
#define AIV_WAIT_MTE3_V()    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3)
#define AIV_SET_V_MTE2()     SetFlag<HardEvent::V_MTE2>(EVENT_ID4)
#define AIV_WAIT_V_MTE2()    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4)
#define AIV_SET_MTE2_MTE3()  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID5)
#define AIV_WAIT_MTE2_MTE3() WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID5)
#define AIV_SET_V_S()        SetFlag<HardEvent::V_S>(EVENT_ID6)
#define AIV_WAIT_V_S()       WaitFlag<HardEvent::V_S>(EVENT_ID6)
#define AIV_SET_S_V()        SetFlag<HardEvent::S_V>(EVENT_ID7)
#define AIV_WAIT_S_V()       WaitFlag<HardEvent::S_V>(EVENT_ID7)
#else
// 回退：SET 侧放一次全栅栏，WAIT 侧空操作 —— 与改造前的语义一致
#define AIV_SET_MTE2_V()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE2_V()    do { } while (0)
#define AIV_SET_V_MTE3()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_MTE3()    do { } while (0)
#define AIV_SET_MTE3_MTE2()  do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE3_MTE2() do { } while (0)
#define AIV_SET_MTE3_V()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE3_V()    do { } while (0)
#define AIV_SET_V_MTE2()     do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_MTE2()    do { } while (0)
#define AIV_SET_MTE2_MTE3()  do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_MTE2_MTE3() do { } while (0)
#define AIV_SET_V_S()        do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_V_S()       do { } while (0)
#define AIV_SET_S_V()        do { PipeBarrier<PIPE_ALL>(); } while (0)
#define AIV_WAIT_S_V()       do { } while (0)
#endif

// ---------------- AIV 侧 UB 布局（字节）----------------
// ⚠ 950 MIX 下 UB 由**一个 AIC + 两个 AIV 子核共享**（同 chunk_fwd_h / KDA fwd_h 的
//   架构约定）：任何"两个子核都会写"的 scratch 必须按 subIdx_ 切成两份，否则会出现
//   跨子核数据竞争（实测表现：h 状态被写坏、且随调度时快时慢 → 概率性错）。
//   只有"按段（off）分区"的缓冲（kBlk/wBlk/vBlk/scr）才可以被子核共享。
//   每个 per-subcore 常量都是**两份的总字节数**，取 Tensor 时按 subIdx_ 偏移一份。
constexpr int32_t PPFM_SUB = 2;      // AIV 子核数（UB 共享）
constexpr int32_t PPFM_SEG = 16;     // left / v_new 的段长（行）
constexpr int32_t PPFM_RB = 32;      // ITER3: 状态更新的行块（行）16->32，
                                      // 每 chunk 状态相位搬运/栅栏减半（UB +24K）

// ---------------- 诊断开关（定位概率性 h 错）----------------
// 打开后：每个工作项把前 N 个 chunk 的 "AIV 读到的 vTmpF_[0]"（AIV 侧）与
// "AIC 写出的 vTmpF_[0]"（AIC 侧）指纹写进 hm 的 m 半边第 0 行（覆盖该行，验收时排除）。
//   lane 0..3  = AIV 读到的 vTmpF_[0]（第 c 个 chunk）
//   lane 8..11 = AIC 写出的 vTmpF_[0]（第 c 个 chunk）
// 判读：两者不等 → AIV 读到别的代（跨核可见性/flag 提前）；相等但≠期望 → AIC 的 mm1 输入不对。
// ⚠ 默认关闭：诊断收尾会把指纹写进 hm 的 m 半边【第 0 行】（见下），
//   验收/对拍脚本不会排除该行 → 默认开启时表现为"m 只有第一行几个元素错、
//   (0,0) 恒为 0"，曾被误判成算子精度缺陷（PPFM-31/33 的 m 坏点就是这么来的）。
//   只在定位跨核可见性时临时打开，并对拍时排除 hm[..., 0, V:]。
#ifndef PPFM_DIAG
#define PPFM_DIAG 0
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
        // P5：列块切分（运行时可配，见 host tiling 的 colSplit）。cb_ = 本工作项负责的列宽，
        // colBase_ = 该列块在整条链里的起始列（h 的 V 列 / m 的 K 列同一个 colBase_）。
        splitNum_ = (t->colSplit > 0) ? static_cast<int32_t>(t->colSplit) : 1;
        cb_ = static_cast<int32_t>(CV_V) / splitNum_;
        colBase_ = 0;
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        hF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_H_F32), CV_K * CV_V);
        mF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_M_F32), CV_K * CV_K);
        hBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_H_BF), CV_K * CV_V);
        mBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_M_BF), CV_K * CV_K);
        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        kBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF_1), CV_BT * CV_K);  // ITER7
        lBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF_1), CV_BT * CV_K);  // ITER7
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
#if PPFM_RD_PROBE
        // 诊断暂存：WS_GATE 区共 4096B，gateF_ 只用前 192 个 float，后面 256 个 float 用来放
        // AIV 读回的 vTmp 行（仅诊断构建使用）。
        // 6 行 × 128 float：row0=vTmp 读回、row2=h 状态 prologue 回读、row3=h 状态读回、
        // row4=dH 读回。
        probeG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_GATE + 1024), 768);
#endif
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
        // ITER10：fixpipe SPLIT_M 把两半写到**同一偏移**（各自 bank），这里用共享基址视图
        vTmpUb_ = ubBuf_.Get<float>()[UB_EXT_F_ELEM];
        stateBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_STATE_BF_ELEM + sstate];
        kBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_KBLK_BF_ELEM];
        wBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_WBLK_BF_ELEM];
        vBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_VBLK_BF_ELEM];
        scrF_ = ubBuf_.Get<float>()[UB_SCR_F_ELEM];
        scrBf_ = ubBuf_.Get<bfloat16_t>()[UB_SCR_BF_ELEM];
#if PPFM_DIAG
        dbgF_ = ubBuf_.Get<float>()[UB_DBG / 4 + subIdx_ * 16];
#endif

        const int64_t taskNum = t->nSeq * t->Hv * static_cast<int64_t>(splitNum_);
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {
            // P5：工作项 = (n, hv, 列块 s)。s 变化最快 ⇒ 同一条链的两个列块尽量落在不同核上。
            const int64_t s = task % static_cast<int64_t>(splitNum_);
            const int64_t hv = (task / static_cast<int64_t>(splitNum_)) % t->Hv;
            const int64_t n = task / (static_cast<int64_t>(splitNum_) * t->Hv);
            colBase_ = static_cast<int32_t>(s) * cb_;
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
            // ITER4：decayF_ 是全 128 项同值 ⇒ 一次 Duplicate 取代 128 次 SetValue
            Duplicate(decayF_, dc, CV_K);
        } else {
#if PPFM_KDA_DECAY_VEC
            // KDA（USE_GK）：decay[k] = 2^(gk_last[k])，整块向量化。
            // 原实现是 128 次 Exp2Scalar：每次都"标量写 UB → 全栅栏 → Exp → 全栅栏 → 标量读"，
            // 实测是 AIV scalar 流水的最大单一来源。这里用 row0F_ 做暂存（本函数里它不承载数据），
            // 逐元素仍是 exp(x·ln2)，与逐点版本逐位等价（L1 位级门禁验证）。
            DataCopy(row0F_, gkGm_[(hv * ctx_.tiling->T + tGlobal) * CV_K], CV_K);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            Muls(row0F_, row0F_, 0.6931471805599453f, CV_K);
            PipeBarrier<PIPE_V>();
            Exp(decayF_, row0F_, CV_K);
#else
            for (int32_t k = 0; k < CV_K; ++k) {
                const float gk = gkGm_.GetValue((hv * ctx_.tiling->T + tGlobal) * CV_K + k);
                decayF_.SetValue(k, Exp2Scalar(gk));
            }
#endif
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessChain(int64_t n, int64_t hv, int64_t bos, int64_t len)
    {
        curN_ = n;
        const auto *t = ctx_.tiling;
#if PPFM_ARCH_IS_950
        // ---- prologue：h = 0（950）----
        Duplicate(row0F_, 0.0f, cb_);
        Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
        PipeBarrier<PIPE_ALL>();   // ITER4：V -> MTE3 只需一次（源行不变）
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            DataCopy(hF32_[r * cb_], row0F_, cb_);
            DataCopy(hBf_[r * cb_], row0Bf_, cb_);
        }
        PipeBarrier<PIPE_ALL>();
#else
        // ---- prologue：h = 0（910B/910_93）----
        // 实测（RD_PROBE 探针）：A2 上用"一次 Cast 出 bf16 行 + 循环内 128 次 MTE3 复用
        // 同一 UB 行"的写法，会把 bf16(h) 落到 GM 时写成 ~1e-3 量级的脏数据（同一循环里的
        // fp32 版本却是严格 0）。AIC 的 mm1 于是把非零的 bf16(h) 当输入，vTmp=W@h≠0，
        // 整条 h 链偏 1.5e-2；m 链因为用逐行写（见下）反而是对的。
        // 这里改成与 m 初值同款的逐行写法：每行重新 Cast，行间用 PIPE_ALL 隔离。
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            Duplicate(row0F_, 0.0f, cb_);
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hF32_[r * cb_], row0F_, cb_);
            DataCopy(hBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#endif
#if PPFM_RD_PROBE
        // 诊断：prologue 写完后立刻回读 h 状态第 0 行（期望全 0）
        if (subIdx_ == 0) {
            DataCopy(row1F_, hF32_, CV_V);           // GM -> UB
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[2 * CV_V], row1F_, CV_V);   // UB -> GM
            PipeBarrier<PIPE_ALL>();
            DataCopy(row2Bf_, hBf_, CV_V);                // bf16(h) 第 0 行（bf16->fp32 观察）
            PipeBarrier<PIPE_ALL>();
            Cast(row2F_, row2Bf_, RoundMode::CAST_NONE, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[1 * CV_V], row2F_, CV_V);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        // m 初值 = I
        // 950：**逐行纯向量构造**（ArithProgression + |k-r| 造对角，省下 64 KiB UB）。
        // ⚠ 不要用 row0F_.SetValue(r,1) 这类"标量写 UB + 向量写同一块 UB"的组合：
        //   实测标量写的落盘顺序不受 PipeBarrier<PIPE_V> 保护，会让个别行丢掉对角 1
        //   （表现为 m 只有 ~0.05% 元素错、max_abs≈1）。
#if PPFM_ARCH_IS_950
        // P5：列块切分后，本工作项只需要列 [colBase_, colBase_+cb_) 的对角
        Duplicate(row2F_, 1.0f, cb_);
        PipeBarrier<PIPE_V>();
        ArithProgression(row1F_, static_cast<float>(colBase_), 1.0f, cb_);   // row1F_[k] = k
        PipeBarrier<PIPE_V>();
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            // ITER4：同一 pipe 内的 4 次 V 运算无需各自栅栏，只在 V->MTE3 与
            //        MTE3 读完（下一轮要覆盖 row0F_/row0Bf_）处各保留一次
            Adds(row0F_, row1F_, -static_cast<float>(r), cb_);   // k - r
            Abs(row0F_, row0F_, cb_);
            Mins(row0F_, row0F_, 1.0f, cb_);
            Sub(row0F_, row2F_, row0F_, cb_);                    // 1 - min(|k-r|,1)
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(mF32_[r * cb_], row0F_, cb_);
            DataCopy(mBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#else
        // 910B/910_93：同一套「ArithProgression + |k-r|」构造在 A2 上**实测退化**——
        //   m 变成"每行常数"（行 r 的值只随 r 变化、整行相同，对角与状态全错；
        //   用 w=0,g=0 探针可复现：m 应为 I，实到 m[r][:] 恒等于 [r%4<2]）。
        //   A2 上 ArithProgression 走 common 实现（标量写 8 拍 + 向量 Add 展开），
        //   与外层逐行向量组合相互干扰；且全仓仅本算子用到该原语（无先例）。
        //   这里改成最朴素、逐行可验证的构造：整行清零 + 单点写 1，
        //   标量写与搬运之间一律用 PIPE_ALL 全栅栏隔离（KDA 的逐点 SetValue 路径
        //   在 A2 上实测正确，说明标量写本身没问题）。
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            Duplicate(row0F_, 0.0f, cb_);
            PipeBarrier<PIPE_ALL>();
            // 只在"本列块包含第 r 列"时写对角 1（P5 列块切分）
            if (r >= colBase_ && r < colBase_ + cb_) {
                row0F_.SetValue(r - colBase_, 1.0f);
            }
            PipeBarrier<PIPE_ALL>();       // S -> V/MTE3
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();       // V -> MTE3
            DataCopy(mF32_[r * cb_], row0F_, cb_);
            DataCopy(mBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();       // MTE3 读完才能下一轮覆盖
        }
#endif
        // 临时诊断（默认关）：给 AIC 即将写的 C 缓冲预置哨兵。
        //   vTmpF_ = 7.0、t1F_ = 5.0 ⇒ 若 AIC 的 fixpipe 正常覆盖，chunk0 的结果不受影响；
        //   若结果里出现 7/5 量级的残留，说明 AIC→AIV 的 C 落点/可见性有问题。
#if PPFM_SENTINEL_PROBE
        Duplicate(scrF_, 7.0f, CV_BT * CV_V);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vTmpF_, scrF_, static_cast<uint32_t>(CV_BT * CV_V));
        Duplicate(scrF_, 5.0f, CV_BT * CV_K);
        PipeBarrier<PIPE_ALL>();
        DataCopy(t1F_, scrF_, static_cast<uint32_t>(CV_BT * CV_K));
        PipeBarrier<PIPE_ALL>();
#endif
        const int64_t nt = (len + CV_BT - 1) / CV_BT;
        // ITER7（P3）：先独立做 chunk 0 的 staging；循环内把 staging(c+1) 提到
        // 「等 dH/T2(c)」之前 ⇒ staging 与 AIC 的 mm2/mm4(c) 重叠（原来 AIV 在这里纯等）
        StageChunk(n, hv, bos, (len < CV_BT) ? len : CV_BT, 0);
        for (int64_t c = 0; c < nt; ++c) {
#if PPFM_DIAG
            curChunk_ = c;
#endif
            const int64_t t0 = bos + c * CV_BT;
            const int64_t left = len - c * CV_BT;
            const int64_t rows = (left < CV_BT) ? left : CV_BT;
            if (c > 0) {
                // dH/T2(c-1) 已在上一轮末尾等到；decayPrevF_ 此时是 decay(c-1)
                ApplyStateUpdates(true, c - 1);
            }
            // ITER5：staging 产物 + 本 chunk 状态一次通知（原来分 kFlagInputs / kFlagState 两次）
            AivSetToAic(kFlagInputs);
            UpdateVNew(hv, t0, rows);
            if (c + 1 < nt) {
                // ITER7：保存本 chunk 的 decay（下一轮推迟状态更新要用），随后 staging 覆盖 decayF_
                Adds(decayPrevF_, decayF_, 0.0f, CV_K);
                PipeBarrier<PIPE_ALL>();
                const int64_t t0n = bos + (c + 1) * CV_BT;
                const int64_t leftn = len - (c + 1) * CV_BT;
                const int64_t rowsn = (leftn < CV_BT) ? leftn : CV_BT;
                StageChunk(n, hv, t0n, rowsn, static_cast<int32_t>((c + 1) & 1));
            }
            AivWaitFromAic(kFlagDH);
            // 读别的核（AIC）写的 GM 前必须让本核缓存行失效，否则会读到过期数据
            // （与 CANN matmul_client.h 中"读跨核 GM flag 前先 DCCI"的用法一致）
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF_);
#endif
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F_);
#endif
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
            // P5：只写本工作项的列块（h 占 [0,V) 列、m 占 [V,V+K) 列，两者同一个 colBase_）
            DataCopy(row0F_, hF32_[r * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + colBase_], row0F_, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(row0F_, mF32_[r * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V + colBase_], row0F_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#if PPFM_RD_PROBE
        // 诊断：把探针各行搬到 hm 的 m 半边第 0..4 行（验收时排除这些行）
        if (n == 0 && hv == 0 && subIdx_ == 0) {
            for (int32_t pr = 0; pr <= 5; ++pr) {
                DataCopy(row0F_, probeG_[pr * CV_V], CV_K);
                PipeBarrier<PIPE_ALL>();
                DataCopy(hmGm_[hmBase + pr * (CV_V + CV_K) + CV_V], row0F_, CV_K);
                PipeBarrier<PIPE_ALL>();
            }
        }
#endif
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
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(diagG_);
#endif
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
    __aicore__ inline void StageChunk(int64_t n, int64_t hv, int64_t t0, int64_t rows, int32_t slot)
    {
        // ITER7：k/left 按 chunk 奇偶写不同 GM 槽，使 staging 可与上一 chunk 的 mm2/mm4 重叠
        GlobalTensor<bfloat16_t> &kOut = ((slot & 1) != 0) ? kBf1_ : kBf_;
        GlobalTensor<bfloat16_t> &lOut = ((slot & 1) != 0) ? lBf1_ : lBf_;
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
        // ITER8（A2）：段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        constexpr int32_t SEG_PER_SUB = (CV_BT / SEG) / PPFM_SUB;
        // 满 chunk 时 AIC 会直接读输入里的 w/k ⇒ 这里不必再 staging 它们
        const bool directInputs = (PPFM_AIC_DIRECT_INPUTS != 0) && (rows == CV_BT);
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            const int32_t valid = (rows > off) ? ((rows - off < SEG) ? (rows - off) : SEG) : 0;
            // P1a：段间复用同一组 UB（kBlk/wBlk/vBlk/scr）。事件语义是"该流水此前所有操作
            // 都完成"，所以在这里成对 set/wait 即可覆盖"上一段的 MTE3 是否读完"，
            // 不需要额外的信用记账。
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            // ITER2：只有尾块需要零填充（整段时下面的 DataCopy 会写满整段）
            if (valid < SEG) {
                Duplicate(kBlkBf_[off * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
                Duplicate(wBlkBf_[off * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
                Duplicate(vBlkBf_[off * cb_], static_cast<bfloat16_t>(0), SEG * cb_);
                PipeBarrier<PIPE_V>();
                AIV_SET_V_MTE2();
                AIV_WAIT_V_MTE2();
            }
            if (valid > 0) {
                DataCopy(kBlkBf_[off * CV_K], kGm_[(hk * t->T + t0 + off) * CV_K],
                         static_cast<uint32_t>(valid * CV_K));
                if (!directInputs) {
                    DataCopy(wBlkBf_[off * CV_K], wGm_[(hv * t->T + t0 + off) * CV_K],
                             static_cast<uint32_t>(valid * CV_K));
                }
                // P5：v 只搬本工作项需要的列窗 [colBase_, colBase_+cb_)（列间隔用 srcStride 跳过）
                DataCopyExtParams vParams{
                    static_cast<uint16_t>(valid),
                    static_cast<uint32_t>(cb_ * static_cast<int32_t>(sizeof(bfloat16_t))),
                    static_cast<uint32_t>((CV_V - cb_) * static_cast<int32_t>(sizeof(bfloat16_t))), 0, 0};
                DataCopyPad(vBlkBf_[off * cb_], vGm_[(hv * t->T + t0 + off) * CV_V + colBase_],
                            vParams, {false, 0, 0, 0});
            }
            AIV_SET_MTE2_MTE3();
            AIV_WAIT_MTE2_MTE3();
            if (!directInputs) {
                DataCopy(kOut[off * CV_K], kBlkBf_[off * CV_K], SEG * CV_K);
                DataCopy(wBf_[off * CV_K], wBlkBf_[off * CV_K], SEG * CV_K);
            }
            // 注意：v 不再落到 GM（v_new 直接从 UB 的 vBlkBf_ 读），省一份 16 KiB/chunk 的 MTE3
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            // left：USE_G 为 bf16(k·dg)，USE_GK 为 k 本身
            if (useG) {
                Cast(scrF_[off * CV_K], kBlkBf_[off * CV_K], RoundMode::CAST_NONE, SEG * CV_K);
                PipeBarrier<PIPE_V>();
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(off + i) * CV_K], scrF_[(off + i) * CV_K], dgF_.GetValue(off + i), CV_K);
                }
                PipeBarrier<PIPE_V>();
                Cast(scrBf_[off * CV_K], scrF_[off * CV_K], RoundMode::CAST_RINT, SEG * CV_K);
                AIV_SET_V_MTE3();
                AIV_WAIT_V_MTE3();   // V -> MTE3
                DataCopy(lOut[off * CV_K], scrBf_[off * CV_K], SEG * CV_K);
            } else {
                DataCopy(lOut[off * CV_K], kBlkBf_[off * CV_K], SEG * CV_K);
            }
            // 段末：本段两次 MTE3（kOut/wBf_ 与 lOut）读完后，下一段才能覆盖对应 UB
        }
        PipeBarrier<PIPE_ALL>();
        // ITER2：dg / decay 落 GM 只是调试用途（只有 PPFM_DIAG 下的 ProbeF32 会读），
        //        默认关掉，每 chunk 省 2 次 DataCopy + 2 次全栅栏
#if PPFM_DIAG
        DataCopy(gateF_[WS_GATE_DG / 4], dgF_, CV_BT);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gateF_[WS_GATE_DECAY / 4], decayF_, CV_K);
        PipeBarrier<PIPE_ALL>();
#endif
        // ITER5：不再在这里发通知——与状态更新合并成一次（见 ProcessChain）
    }

    __aicore__ inline void UpdateVNew(int64_t hv, int64_t t0, int64_t rows)
    {
        const bool useG = (ctx_.tiling->gateMode == PPFM_GATE_USE_G);
        AivWaitFromAic(kFlagHalf1);
#if PPFM_VTMP_UB_DIAG
        if (curChunk_ == 1) {
            PipeBarrier<PIPE_ALL>();
            const float g0 = vTmpF_.GetValue(0);
            const float g32 = vTmpF_.GetValue(32 * CV_V);
            const float uOwn0 = extBlkF_.GetValue(0);
            const float uOwn32 = extBlkF_.GetValue(32 * CV_V);
            const float uSh0 = vTmpUb_.GetValue(0);
            const float uSh32 = vTmpUb_.GetValue(32 * CV_V);
            PipeBarrier<PIPE_ALL>();
            row0F_.SetValue(0, g0);
            row0F_.SetValue(1, g32);
            row0F_.SetValue(2, uOwn0);
            row0F_.SetValue(3, uOwn32);
            row0F_.SetValue(4, uSh0);
            row0F_.SetValue(5, uSh32);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[((curN_ * ctx_.tiling->Hv + hv) * CV_K) * (CV_V + CV_K) + CV_V],
                     row0F_, 8);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        // ITER9：vTmp 走 GM 时保留过渡探读（A2 走 UB 时才省掉）
#if PPFM_LEGACY_PROBE_READS
#if !PPFM_VTMP_UB
        DataCopy(row2F_, vTmpF_, 8);
        PipeBarrier<PIPE_ALL>();
#endif
        DataCopy(row2F_, t1F_, 8);
        PipeBarrier<PIPE_ALL>();
#endif  // PPFM_LEGACY_PROBE_READS
#if PPFM_DIAG
        // 诊断：记下"本子核读到的 vTmpF_[0]"（chunk 0 时它必须恰好是 0）
        if (curChunk_ < PPFM_DIAG_CHUNKS) {
            PipeBarrier<PIPE_ALL>();
            dbgF_.SetValue(static_cast<int32_t>(curChunk_), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
#if !PPFM_VTMP_UB
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
        // v_new = (v - vTmp) · dg → bf16（逐行；整块版本会引入 ~0.4% 的 GDN 偏差，待查）
        // v_new = (v - vTmp)·dg → bf16：同样**按整段分配 subcore**，段内做完 Cast/Sub/缩放/Cast/落盘
        constexpr int32_t SEG = 16;
        // ITER8（A2）：段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        constexpr int32_t SEG_PER_SUB = (CV_BT / SEG) / PPFM_SUB;
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            const int32_t lo = (seg - subIdx_ * SEG_PER_SUB) * SEG;
#if !PPFM_VTMP_UB
            // A5 主线：vTmp 仍从 GM 回读
            DataCopy(extBlkF_[lo * cb_], vTmpF_[off * cb_], SEG * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
#if PPFM_RD_PROBE
        // 诊断：把本子核读到的 vTmp 第 0 行原样存到 GM 暂存（首个 chunk，子核 0）
            if (probeCnt_ == 0 && subIdx_ == 0) {
                DataCopy(probeG_, extBlkF_, cb_);
                PipeBarrier<PIPE_ALL>();
                probeCnt_ = 1;
            }
#endif
#endif
            Cast(scrF_[off * cb_], vBlkBf_[off * cb_], RoundMode::CAST_NONE, SEG * cb_);
            PipeBarrier<PIPE_V>();
#if PPFM_VTMP_UB
            // ITER10：从共享基址视图读本子核那半（lo ∈ {0, SEG}）
            Sub(scrF_[off * cb_], scrF_[off * cb_], vTmpUb_[lo * cb_], SEG * cb_);
#else
            Sub(scrF_[off * cb_], scrF_[off * cb_], extBlkF_[lo * cb_], SEG * cb_);
#endif
            PipeBarrier<PIPE_V>();
            if (useG) {
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(off + i) * cb_], scrF_[(off + i) * cb_], dgF_.GetValue(off + i), cb_);
                }
                PipeBarrier<PIPE_V>();
            }
            Cast(scrBf_[off * cb_], scrF_[off * cb_], RoundMode::CAST_RINT, SEG * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(vNewBf_[off * cb_], scrBf_[off * cb_], SEG * cb_);
        }
        // 各子核只写自己那两段（off 互不相交），循环内无需段间信用；
        // 出口保留一次全栅栏，供后面的 staging 复用 scrF_/scrBf_（跨函数边界）。
        PipeBarrier<PIPE_ALL>();
        // bf16(T1)：同样按段分配
        // ITER8（A2）：段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        // ITER8（A2）：SEG_PER_SUB 已在 v_new 循环前声明（同一函数内不能重复定义）
#if !PPFM_T1_FIXPIPE_BF16
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            DataCopy(scrF_[off * cb_], t1F_[off * cb_], SEG * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            Cast(scrBf_[off * cb_], scrF_[off * cb_], RoundMode::CAST_RINT, SEG * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(t1Bf_[off * cb_], scrBf_[off * cb_], SEG * cb_);
        }
        PipeBarrier<PIPE_ALL>();
#endif
#if PPFM_RD_PROBE
        // 诊断：vNewBf_ 第 0 行（bf16→fp32）读回，确认 AIV 写出的 B 内容
        if (probeCnt_ == 1 && subIdx_ == 0) {
            DataCopy(row2Bf_, vNewBf_, CV_V);
            PipeBarrier<PIPE_ALL>();
            Cast(row2F_, row2Bf_, RoundMode::CAST_NONE, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[5 * CV_V], row2F_, CV_V);
            PipeBarrier<PIPE_ALL>();
        }
#endif
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
        // 同 UpdateVNew 的过渡探读：dH / T2 也是 AIC 刚写、本核刚读的 GM
#if PPFM_LEGACY_PROBE_READS
        DataCopy(row2F_, dhBuf, 8);
        PipeBarrier<PIPE_ALL>();
        DataCopy(row2F_, t2Buf, 8);
        PipeBarrier<PIPE_ALL>();
#endif  // PPFM_LEGACY_PROBE_READS
        // 每 RB 行一次搬运：块内逐行 Muls（廉价、无需栅栏），块级 Add/Sub/Cast
        constexpr int32_t RB = PPFM_RB;   // ITER6a：真正用上 32 行（ITER3 只改了 UB 尺寸）
        for (int32_t rb = subIdx_ * RB; rb < CV_K; rb += subNum_ * RB) {
            // ITER2：第一次搬运后的全栅栏冗余——紧随其后第二次搬运之后还有一次，
            //        足以保证两次 MTE2 都在 Muls/Add 之前完成
            // P1a：上一轮的 MTE3（写 hF32_/hBf_）读的是同一组 UB，先等它读完再覆盖
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            DataCopy(stateBlkF_, hF32_[rb * cb_], RB * cb_);
            DataCopy(extBlkF_, dhBuf[rb * cb_], RB * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
#if PPFM_RD_PROBE
            // 诊断：首个 RB 块里，把"读到的 h 状态"和"读到的 dH"各留一行
            if (rb == subIdx_ * RB && subIdx_ == 0) {
                DataCopy(probeG_[3 * CV_V], stateBlkF_, CV_V);   // h 状态读回
                DataCopy(probeG_[4 * CV_V], extBlkF_, CV_V);     // dH 读回
                PipeBarrier<PIPE_ALL>();
            }
#endif
            if (useG) {
                // GDN：每 chunk 一个标量 decay ⇒ 整块一次 Muls（原来 16 次逐行 Muls +
                // 16 次 GetValue；h/m 合计每 chunk 每子核 128 次，是 AIV SCALAR 的主要来源）
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(stateBlkF_, stateBlkF_, dc, RB * cb_);
            } else {
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(stateBlkF_[(r - rb) * cb_], stateBlkF_[(r - rb) * cb_], dc, cb_);
                }
            }
            PipeBarrier<PIPE_V>();
            Add(stateBlkF_, stateBlkF_, extBlkF_, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(hF32_[rb * cb_], stateBlkF_, RB * cb_);
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();   // 上一次 MTE3 读完 stateBlkF_/stateBlkBf_ 才能覆盖
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(hBf_[rb * cb_], stateBlkBf_, RB * cb_);
        }
        PipeBarrier<PIPE_ALL>();
        for (int32_t rb = subIdx_ * RB; rb < CV_K; rb += subNum_ * RB) {
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            DataCopy(stateBlkF_, mF32_[rb * cb_], RB * cb_);
            DataCopy(extBlkF_, t2Buf[rb * cb_], RB * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            if (useG) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(stateBlkF_, stateBlkF_, dc, RB * cb_);
            } else {
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(stateBlkF_[(r - rb) * cb_], stateBlkF_[(r - rb) * cb_], dc, cb_);
                }
            }
            PipeBarrier<PIPE_V>();
            Sub(stateBlkF_, stateBlkF_, extBlkF_, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(mF32_[rb * cb_], stateBlkF_, RB * cb_);
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(mBf_[rb * cb_], stateBlkBf_, RB * cb_);
        }
        PipeBarrier<PIPE_ALL>();
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
    LocalTensor<float> vTmpUb_;   // ITER10（A2）：共享基址的 vTmp 落点视图
    LocalTensor<bfloat16_t> stateBlkBf_;
    LocalTensor<bfloat16_t> kBlkBf_;
    LocalTensor<bfloat16_t> wBlkBf_;
    LocalTensor<bfloat16_t> vBlkBf_;
    LocalTensor<float> scrF_;
    LocalTensor<bfloat16_t> scrBf_;
    int32_t subIdx_ = 0;
    int32_t subNum_ = 1;
    int32_t splitNum_ = 1;   // P5 列块数（1 或 2），由 tiling.colSplit 决定
    int32_t cb_ = CV_V;      // P5 本工作项的列宽（128 或 64）
    int32_t colBase_ = 0;    // P5 本工作项列块在整条链里的起始列
    int64_t curN_ = 0;   // ITER11：UB 诊断探针要定位本链 hm 地址
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
    GlobalTensor<bfloat16_t> kBf1_;  // ITER7
    GlobalTensor<bfloat16_t> lBf1_;  // ITER7
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
#if PPFM_RD_PROBE
    GlobalTensor<float> probeG_;     // 诊断：AIV 读回的 vTmp 行暂存
    int32_t probeCnt_ = 0;
#endif
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
        // P5：列块切分（与 AIV 侧同一套解码；cb_/colBase_ 见 PpFwdVector::Run）
        splitNum_ = (t->colSplit > 0) ? static_cast<int32_t>(t->colSplit) : 1;
        cb_ = static_cast<int32_t>(CV_V) / splitNum_;
        colBase_ = 0;
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        kBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF_1), CV_BT * CV_K);  // ITER7
        lBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF_1), CV_BT * CV_K);  // ITER7
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
        // PPFM_AIC_DIRECT_INPUTS：满 chunk 时直接读输入张量里的 w/k（省掉 AIV 的 staging）
        wIn_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.w));
        kIn_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.k));
        const int64_t taskNum = t->nSeq * t->Hv * static_cast<int64_t>(splitNum_);
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {
            const int64_t s = task % static_cast<int64_t>(splitNum_);
            const int64_t hv = (task / static_cast<int64_t>(splitNum_)) % t->Hv;
            const int64_t n = task / (static_cast<int64_t>(splitNum_) * t->Hv);
            colBase_ = static_cast<int32_t>(s) * cb_;
            const int64_t bos = cuGm_.GetValue(n);
            const int64_t eos = cuGm_.GetValue(n + 1);
            const int64_t len = eos - bos;
            const int64_t nt = (len + CV_BT - 1) / CV_BT;
            for (int64_t c = 0; c < nt; ++c) {
                const int64_t leftLen = len - c * CV_BT;
                const int64_t rows = (leftLen < CV_BT) ? leftLen : CV_BT;
                ProcessChunk(c, hv, bos + c * CV_BT, rows);
            }
        }
    }

private:
    __aicore__ inline void ProcessChunk(int64_t c, int64_t hv, int64_t t0, int64_t rows)
    {
        // dH / T2 双缓冲：本 chunk 写到自己那一份（不用拷贝，直接分支）
        const bool evenChunk = ((c & 1) == 0);
        GlobalTensor<float> &dhBuf = evenChunk ? dHF_ : dHF1_;
        GlobalTensor<float> &t2Buf = evenChunk ? t2F_ : t2F1_;
        // 满 chunk 时 w/k 直接用输入张量（AIV 侧不再 staging）；尾块仍用零填充后的 staging
        const auto *tt = ctx_.tiling;
        const int64_t hk = hv / (tt->hvPerHk == 0 ? 1 : tt->hvPerHk);
        const bool directInputs = (PPFM_AIC_DIRECT_INPUTS != 0) && (rows == CV_BT);
        GlobalTensor<bfloat16_t> wTile =
            directInputs ? wIn_[(hv * tt->T + t0) * CV_K] : wBf_;
        // ① vTmp[BT,V] = W_c[BT,K] @ bf16(h)[K,V]
        // ITER5：inputs 与 state 已合并为同一次通知
        AicWaitFromAiv(kFlagInputs);
        // ⚠ 读别的核（AIV）刚写过的 GM 之前必须让本核的 cache 失效：h/m 每 chunk 都被
        //    AIV 重写，若 AIC 命中自己缓存的旧行，mm1/mm3 就会拿到过期的 h/m
        //    （实测表现为概率性的 h 半边大面积错、m 只是略偏，且随调度时好时坏）。
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(wBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(hBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(mBf_);
#endif
#if PPFM_TILE_MMAD
        RunTiledNT(wTile, hBf_, vTmpF_, CV_BT, static_cast<uint32_t>(cb_), CV_K,
                    /*toUb=*/(PPFM_VTMP_UB != 0));   // ITER9：按宏选择 A2 UB 落点
#else
        RunMmadNT(wTile, hBf_, vTmpF_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#endif

        // ③ T1[BT,K] = W_c[BT,K] @ bf16(m)[K,K]
#if PPFM_TILE_MMAD
#if PPFM_T1_FIXPIPE_BF16
        // fixpipe 直接把 T1 量化成 bf16 落 t1Bf_（mm4 的 B 操作数），AIV 侧不再往返
        RunTiledNT(wTile, mBf_, t1Bf_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#else
        RunTiledNT(wTile, mBf_, t1F_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#endif
#else
        RunMmadNT(wTile, mBf_, t1F_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#endif
        // ⚠ 写侧也要 clean（写回），只靠读者 DCCI 不够：FIX 写回可能还停在写缓冲里，
        //   此时 AIV 即便 DCCI 也会读到旧值。实测（PPFM_DIAG 指纹）：AIV 读到 vTmp 全 0，
        //   而 AIC 实际写了 -0.013/+0.0092 → v_new 退化成 v，h 半边随机整头崩（m 不受影响）。
#if !PPFM_VTMP_UB
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
        // ⚠ 正式原语：DDR 数据同步屏障——保证 C 的写回对其他核可见后再抬 flag。
        //   诊断版实验表明竞态是"flag 已到、写回仍在途"的时序窗口（加探针即掩盖）。
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        AicSetToAiv(kFlagHalf1);

        // ② dH[K,V] = k_c^T @ bf16(v_new)[BT,V]
        AicWaitFromAiv(kFlagVNew);
        // ITER7：按 chunk 奇偶取 k/left 槽（与 AIV staging 写入槽一致）
        GlobalTensor<bfloat16_t> &kIn = ((c & 1) != 0) ? kBf1_ : kBf_;
        GlobalTensor<bfloat16_t> &lIn = ((c & 1) != 0) ? lBf1_ : lBf_;
        // 满 chunk：k 也直接来自输入（AIV 不再写 kBf_）
        GlobalTensor<bfloat16_t> kTile = directInputs ? kIn_[(hk * tt->T + t0) * CV_K] : kIn;
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(kIn);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(lIn);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vNewBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1Bf_);
#endif
        RunMmadTA(kTile, vNewBf_, dhBuf, CV_K, static_cast<uint32_t>(cb_), CV_BT);

        // ④ T2[K,K] = left^T @ bf16(T1)[BT,K]
        RunMmadTA(lIn, t1Bf_, t2Buf, CV_K, static_cast<uint32_t>(cb_), CV_BT);
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(dhBuf);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t2Buf);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        // dH 与 T2 都由 AIV 在**下一个 chunk 开头**使用，合并为一次跨核通知（省一次 flag 往返）
        AicSetToAiv(kFlagDH);
#if PPFM_DIAG
        if (c < PPFM_DIAG_CHUNKS) {
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
            PipeBarrier<PIPE_ALL>();
            diagG_.SetValue(static_cast<int32_t>(c), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
    }

    // A 行主：C[m,n] = A[m,k] @ B[k,n]（A/B 都是 bf16、行主；C 是 fp32 行主）
#if PPFM_TILE_MMAD
    // ---- 手写 tile 级（增量 A1）：GM→L1→L0A/L0B→MMAD→C 回写（落点仍是 GM）----
    // 目的：先用与 BlockMmad 相同的落点验证 tile/MMAD 数值一致；A5 的 L0C→UB 在 A2 增量里接。
    // ITER8（A2）：toUb=true 时 C 落 UB_EXT_F 区的共享槽（fixpipe SPLIT_M），否则仍落 gmC
    // CT = C 的元素类型：float（默认）或 bfloat16_t（fixpipe 直接按输入 dtype 量化，省掉
    // AIV 侧的"读回 fp32 → Cast → 写 bf16"整条回路，见 PPFM_T1_FIXPIPE_BF16）。
    template <class CT>
    __aicore__ inline void RunTiledNT(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                      GlobalTensor<CT> &gmC, uint32_t m, uint32_t n, uint32_t k,
                                      bool toUb = false)
    {
        Catlass::Arch::Resource<MmArchTag> res;
        auto l1A = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_A_OFF);
        auto l1B = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_B_OFF);
        auto l0A = res.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0B = res.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0C = res.l0CBuf.template GetBufferByByte<float>(0);

        auto tA = tla::MakeTensor(gmA[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k),
                                  Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n),
                                  Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], tla::MakeLayout<CT, Catlass::layout::RowMajor>(m, n),
                                  Catlass::Arch::PositionGM{});
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        auto tL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),
            Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),
            Catlass::Arch::PositionL1{});
        typename MmTileCopyNT::template CopyGmToL1A<decltype(bA)> copyG2LA;
        typename MmTileCopyNT::template CopyGmToL1B<decltype(bB)> copyG2LB;
        copyG2LA(tL1A, bA);
        copyG2LB(tL1B, bB);
        // 跨流水必须用事件对（chunk_fwd_h_cube.h 的写法），PIPE_ALL 不保证 MTE1/M/FIX 次序
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        auto tL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        typename MmTileCopyNT::CopyL1ToL0A copyL2L0A;
        typename MmTileCopyNT::CopyL1ToL0B copyL2L0B;
        copyL2L0A(tL0A, GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k)));
        copyL2L0B(tL0B, GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        auto tL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        MmTileMmadNT mmad;
        mmad(tL0C, tL0A, tL0B, m, n, k);
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);

        if constexpr (std::is_same_v<CT, float>) {
            // fp32 C：950 可直接 L0C→UB（SPLIT_M），A2/A3 只能落 GM
            bool handled = false;
#if PPFM_ARCH_IS_950
            if (toUb) {
                // ITER8（A2）：写进 UB_EXT_F 区（64x128 fp32 = 32KB，正好是该区尺寸）。
                // SPLIT_M 语义：整块的「前一半行」落在该地址的低半区、「后一半行」落高半区，
                // 与「段按连续半区分配 subcore」对齐 ⇒ 两个子核各读自己那半。
                AscendC::LocalTensor<float> vTmpUb(AscendC::TPosition::VECCALC, UB_EXT_F, CV_BT * CV_V);
                auto layoutUb = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
                auto tensorUb = tla::MakeTensor(vTmpUb, layoutUb, Catlass::Arch::PositionUB{});
                typename TiledCopyNTSplitUb::template CopyL0CToDst<decltype(tensorUb)> copyUb;
                copyUb(tensorUb, tL0C);
#if PPFM_VTMP_UB_DIAG
                typename MmTileCopyNT::template CopyL0CToDst<decltype(bC)> copyCRef;
                copyCRef(bC, tL0C, static_cast<uint8_t>(0));   // 诊断参照：同值再落一份 GM
#endif
                handled = true;
            }
#endif
            if (!handled) {
                MmCopyL0CToGm<decltype(bC)> copyC;
                // ⚠ 必须走 3 参重载 (dst, src, unitFlag)：4 参会误选 (l0Batch, dstNdStride)
                //    批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，C 恒为初值 0。
                copyC(bC, tL0C, static_cast<uint8_t>(0));
            }
        } else {
            // bf16 C（PPFM_T1_FIXPIPE_BF16）：fixpipe 直接把 fp32 的 C 量化成 bf16 落 GM，
            // AIV 侧不再需要"读回 fp32 → Cast → 写 bf16"。
            MmCopyL0CToGm<decltype(bC)> copyC;
            copyC(bC, tL0C, static_cast<uint8_t>(0));
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }

    // A 列主（A 在 GM 上是 [k,m] 列主，逻辑 [m,k]）
    __aicore__ inline void RunTiledTA(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                      GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> res;
        auto l1A = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_A_OFF);
        auto l1B = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_B_OFF);
        auto l0A = res.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0B = res.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0C = res.l0CBuf.template GetBufferByByte<float>(0);

        auto tA = tla::MakeTensor(gmA[0], tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k),
                                  Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n),
                                  Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n),
                                  Catlass::Arch::PositionGM{});
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        auto tL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),
            Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),
            Catlass::Arch::PositionL1{});
        typename MmTileCopyTA::template CopyGmToL1A<decltype(bA)> copyG2LA;
        typename MmTileCopyTA::template CopyGmToL1B<decltype(bB)> copyG2LB;
        copyG2LA(tL1A, bA);
        copyG2LB(tL1B, bB);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        auto tL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        typename MmTileCopyTA::CopyL1ToL0A copyL2L0A;
        typename MmTileCopyTA::CopyL1ToL0B copyL2L0B;
        copyL2L0A(tL0A, GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k)));
        copyL2L0B(tL0B, GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        auto tL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        MmTileMmadTA mmad;
        mmad(tL0C, tL0A, tL0B, m, n, k);
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);

        MMTACopyL0CToGm<decltype(bC)> copyC;
        // ⚠ 必须走 3 参重载 (dst, src, unitFlag)：4 参会误选 (l0Batch, dstNdStride)
        //    批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，C 恒为初值 0。
        copyC(bC, tL0C, static_cast<uint8_t>(0));
        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }
#endif  // PPFM_TILE_MMAD

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
    int32_t splitNum_ = 1;   // P5 列块数（1 或 2），与 AIV 侧同源（tiling.colSplit）
    int32_t cb_ = CV_V;      // P5 本工作项的列宽（128 或 64）
    int32_t colBase_ = 0;    // P5 本工作项列块起始列（AIC 侧只用于诊断/一致性）
    GlobalTensor<bfloat16_t> wIn_;   // PPFM_AIC_DIRECT_INPUTS：输入 w 视图
    GlobalTensor<bfloat16_t> kIn_;   // PPFM_AIC_DIRECT_INPUTS：输入 k 视图
    GlobalTensor<bfloat16_t> kBf1_;  // ITER7
    GlobalTensor<bfloat16_t> lBf1_;  // ITER7
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
