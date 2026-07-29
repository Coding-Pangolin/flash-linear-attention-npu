/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaBwdWyDqkgFused — shared definitions (BNSD + optional TND varlen, GVA).
 *
 * Math reference: Triton `chunk_kda_bwd_kernel_wy_dqkg_fused`
 *   (flash-linear-attention/fla/ops/kda/chunk_bwd.py L124-296).
 *
 * Stage graph (per chunk `i_t`, head `i_hv`; `i_h = i_hv // (HV/H)`):
 *   Stage0 WyV   (once)   : Cube  dA += dv@v^T ; dvb = A@dv        -> Vec dv2/db
 *   Stage1 KvAcc (per BK) : Cube  dq/dk/dw += do@h / v_new@dh / dv@h (sum over V-tiles)
 *                           Vec (parallel, no cube dependency) kg = k*exp2(g); dgk = colsum(h*dh)*exp2(gn)
 *   Stage2 GateWy(per BK) : Vec   gate dq/dk, dw=-dw, dgk           -> Cube dA += dw@kg^T ; dkgb = A@dw
 *                           Vec   epilog: db, dg, dk += dkgb*gk*beta ; store dq/dk/dg
 *   Stage3 DaFinal(once)  : Vec mask dA*=beta[j] (i>j)  -> Cube dA@A ; A@dA -> Vec negate + store dA/db
 *
 * MIX_AIC_1_2. Chunk-independent parallel across cores; HV serial in-core.
 * Pipelining: 4 GM workspace slots cycled per (task,hv) unit, raw CrossCore 0x2 credit
 * semantics (multiple Set/Wait pairs per hv are safe because AIC/AIV loop hv and the BK
 * sub-loop in lockstep order — see cube.h / vector.h).
 *
 * TilingData: NOT included from op_host here (kernel picks it up via GET_TILING_DATA from
 * whatever op_host registers under the name ChunkKdaBwdWyDqkgFusedTilingData). This file
 * assumes the following field names (per task spec / host contract):
 *   batch, t, h, hv, k, v, chunkSize, numChunks, totalTasks, hasCuSeqlens, hasChunkIndices,
 *   seqNum, dataType, usedCoreNum, wsBytes, wsSlotBytes, scale, stateVFirst, bk, bv
 * `bk`/`bv` are treated as upper-bound hints; the kernel internally retiles K into
 * ceil(K/kMAX_BK) tiles and V into ceil(V/MAX_BV) tiles to respect UB budget regardless of
 * the host-provided values.
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_COMMON_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_COMMON_H

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;

namespace kda_wy_dqkg {

constexpr float LN2 = 0.6931471805599453f;
constexpr float EXP2_CLAMP = 80.0f;
constexpr float EXP_INPUT_MAX = EXP2_CLAMP * LN2;
constexpr float EXP_INPUT_MIN = -EXP2_CLAMP * LN2;
constexpr float FP16_MAX = 65504.0f;

// Compile-time tile caps: internal retiling regardless of host bk/bv hints (UB-budget driven).
constexpr uint32_t MAX_BT = 64;
constexpr uint32_t MAX_BK = 64;   // K retiled into ceil(K/MAX_BK) tiles (K<=128 -> <=2 tiles)
// V retiled into ceil(V/MAX_BV) tiles. Cap at 64 so AIV UB stays under AtlasA2 192KB:
// Gate peak needs ~6*BT*BK + 2*BK*BV fp32 plus T scratch; BV=128 overflows.
constexpr uint32_t MAX_BV = 64;   // V<=128 -> 2 tiles; V<=256 -> 4 tiles
constexpr uint32_t MAX_K_TOTAL = 128;
constexpr uint32_t MAX_V_TOTAL = 256;
constexpr uint32_t MAX_NBK = (MAX_K_TOTAL + MAX_BK - 1) / MAX_BK;   // 2
constexpr uint32_t MAX_NBV = (MAX_V_TOTAL + MAX_BV - 1) / MAX_BV;   // 4

constexpr uint32_t NUM_GM_SLOTS = 4;

// Pipeline opt switches (baseline path ships with all 0 except 4-slot windowing).
// Flip after precision lands; see README / DESIGN §流水.
#ifndef USE_L1_A_RESIDENT
#define USE_L1_A_RESIDENT 1
#endif
#ifndef USE_STAGE1_L0C_ACCUM
#define USE_STAGE1_L0C_ACCUM 1
#endif
#ifndef USE_GATE_REUSE_KG_WS
#define USE_GATE_REUSE_KG_WS 1
#endif
#ifndef USE_EPILOG_VEC_FOLD
#define USE_EPILOG_VEC_FOLD 1
#endif
#ifndef USE_DUAL_AIV_MASK
#define USE_DUAL_AIV_MASK 1
#endif
#ifndef USE_EARLY_MASK_PER_HEAD
#define USE_EARLY_MASK_PER_HEAD 1
#endif
#ifndef USE_STAGE2_PRELOAD_A
#define USE_STAGE2_PRELOAD_A 1
#endif
#ifndef USE_STAGE0_DA_L0C_ACCUM
#define USE_STAGE0_DA_L0C_ACCUM 1
#endif
#ifndef USE_GATE_EARLY_SET
#define USE_GATE_EARLY_SET 1
#endif
#ifndef USE_DUAL_AIV_STORE
#define USE_DUAL_AIV_STORE 1
#endif
#ifndef USE_MASK_SOFT_LEAD
#define USE_MASK_SOFT_LEAD 1
#endif
// I5: Prefill Stage0 for 2 windows; steady Post(w) then Stage0(w+2).
// Cube WaitFree only before Stage0 on a reused bank (after Post of that bank).
#ifndef USE_WIN_SOFT_LEAD
#define USE_WIN_SOFT_LEAD 1
#endif
#ifndef KDA_BWD_PREFILL_WINDOWS
#define KDA_BWD_PREFILL_WINDOWS 2
#endif
#ifndef USE_L0_AB_DBUF
#define USE_L0_AB_DBUF 0
#endif
// Model-scale still trips FIXP/L0C ECC with overlap; keep off until L1-resident /
// Preload races are fully audited. State machine (fixMte2Outstanding + evt=14) is fixed.
#ifndef USE_FIX_MTE2_OVERLAP
#define USE_FIX_MTE2_OVERLAP 0
#endif
// Tried: per-head Kg→Gate. Regressed +0.06ms vs I5 (7333 vs 7271) — keep off.
#ifndef USE_KG_GATE_INTERLEAVE
#define USE_KG_GATE_INTERLEAVE 0
#endif
// Tried: one V_S0 per window. Regressed +0.12ms (7375 vs 7259) — keep off.
#ifndef USE_VS0_ONCE_PER_WINDOW
#define USE_VS0_ONCE_PER_WINDOW 0
#endif
// Tried: park exp(gn) in Kg. +0.06ms vs state-panel (6029 vs 5971) — keep off.
#ifndef USE_EXP_GN_PARK
#define USE_EXP_GN_PARK 0
#endif
// P1: PR190-style MTE2∥V ping-pong on Gate/Epilog loads (AllocEventID 2-slot).
#ifndef USE_VEC_MTE2_PP
#define USE_VEC_MTE2_PP 1
#endif
#ifndef USE_VEC_MTE2_PP_EPILOG
#define USE_VEC_MTE2_PP_EPILOG 0
#endif

// Cross-core flags (AIC <-> AIV), raw counting semantics (0x2). Re-used across BK
// sub-iterations and across (task,hv) units — Set/Wait counts match because both
// sides iterate task/hv/BK in the same order.
//
// FFTS encodes flag_id in 4 bits (flag_id & 0xf). IDs MUST be unique in 0..15.
// Skip 8/9/10: Catlass reserves them for AIV/AIC inter-block / AIV subblock barriers.
constexpr uint8_t FLAG_C_S0 = 0;       // AIC -> AIV: stage0 dA/dvb ready
constexpr uint8_t FLAG_C_S1 = 1;       // AIC -> AIV: stage1 dq/dk/dw partial sums ready
constexpr uint8_t FLAG_V_GATE = 2;     // AIV -> AIC: stage2 gated dw/kg ready
constexpr uint8_t FLAG_C_S2 = 3;       // AIC -> AIV: stage2 dA delta / dkgb ready
constexpr uint8_t FLAG_V_MASK = 4;     // AIV -> AIC: stage3 masked dA ready
constexpr uint8_t FLAG_C_S3 = 5;       // AIC -> AIV: stage3 dA@A@A ready
constexpr uint8_t FLAG_V_S0 = 6;       // AIV -> AIC: Stage0Vec done (before Cube Stage1)
constexpr uint8_t FLAG_SLOT_FREE0 = 7;
constexpr uint8_t FLAG_SLOT_FREE1 = 11;
constexpr uint8_t FLAG_SLOT_FREE2 = 12;
constexpr uint8_t FLAG_SLOT_FREE3 = 13;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif

// ---------------------------------------------------------------------------
// Workspace slot layout (element counts), per slot. fp32 region and T (qk-dtype)
// region are laid out as two separate byte ranges within the slot.
// ---------------------------------------------------------------------------
struct SlotLayoutF32 {
    // Stage0
    static constexpr uint64_t dAWs = 0;                                   // [BT,BT]
    static constexpr uint64_t dASlot = dAWs + MAX_BT * MAX_BT;            // [NBV][BT,BT]
    // Contiguous V-tile panels [NBV][BT,BV] — avoid FixPipe writes into strided
    // [BT,V_TOTAL] columns (partial BT + 2nd V-tile left some rows zero/wrong).
    static constexpr uint64_t dvbWs = dASlot + MAX_NBV * MAX_BT * MAX_BT; // [NBV][BT,MAX_BV]
    // Stage1 (per-BK, per-v-tile deltas)
    static constexpr uint64_t dqSlot = dvbWs + MAX_BT * MAX_V_TOTAL;      // [NBV][BT,BK]
    static constexpr uint64_t dkSlot = dqSlot + MAX_NBV * MAX_BT * MAX_BK;
    static constexpr uint64_t dwSlot = dkSlot + MAX_NBV * MAX_BT * MAX_BK;
    // Stage2
    static constexpr uint64_t dADeltaWs = dwSlot + MAX_NBV * MAX_BT * MAX_BK; // [BT,BT]
    static constexpr uint64_t dkPartialWs = dADeltaWs + MAX_BT * MAX_BT;      // [BT,BK]
    static constexpr uint64_t gkWs = dkPartialWs + MAX_BT * MAX_BK;           // [BT,BK]
    static constexpr uint64_t dkgbWs = gkWs + MAX_BT * MAX_BK;                // [BT,BK]
    static constexpr uint64_t dgkWs = dkgbWs + MAX_BT * MAX_BK;               // [BK]
    // Stage3
    static constexpr uint64_t dA3Ws = dgkWs + MAX_BK;   // [BT,BT]
    // Misc / AIV merge (subBlock0/1 partials)
    static constexpr uint64_t betaWs = dA3Ws + MAX_BT * MAX_BT; // [BT] (unused legacy)
    static constexpr uint64_t dbMergeWs = betaWs + MAX_BT;      // [2][BT]
    static constexpr uint64_t dgkMergeWs = dbMergeWs + 2 * MAX_BT; // [2][BK]
    // Gated dq parked here so Cube Stage1 next BK can overwrite dqSlot safely.
    static constexpr uint64_t dqGatedWs = dgkMergeWs + 2 * MAX_BK; // [BT,BK]
    // Kg→Gate park: avoid Gate reloading GM k/g (USE_GATE_REUSE_KG_WS).
    static constexpr uint64_t kParkWs = dqGatedWs + MAX_BT * MAX_BK; // [BT,BK]
    static constexpr uint64_t gParkWs = kParkWs + MAX_BT * MAX_BK;   // [BT,BK]
    static constexpr uint64_t TOTAL = gParkWs + MAX_BT * MAX_BK;
};

struct SlotLayoutT {
    static constexpr uint64_t kgWs = 0;                        // [BT,BK]
    static constexpr uint64_t dwNegWs = kgWs + MAX_BT * MAX_BK;
    static constexpr uint64_t dAMaskedWs = dwNegWs + MAX_BT * MAX_BK; // [BT,BT]
    static constexpr uint64_t dA2InterimWs = dAMaskedWs + MAX_BT * MAX_BT;
    // stateVFirst Stage1 packs: contiguous [NBV][BV,BK] panels (not aliased with Stage3).
    static constexpr uint64_t stateHWs = dA2InterimWs + MAX_BT * MAX_BT;
    static constexpr uint64_t stateDhWs = stateHWs + MAX_NBV * MAX_BV * MAX_BK;
    static constexpr uint64_t TOTAL = stateDhWs + MAX_NBV * MAX_BV * MAX_BK;
};

// ---------------------------------------------------------------------------
// Shared base: GM buffers, tiling scalars, addressing helpers.
// ---------------------------------------------------------------------------
template <typename T>
class KdaWyDqkgBase {
public:
    __aicore__ inline void InitCommon(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR g, GM_ADDR beta,
                                      GM_ADDR a, GM_ADDR h, GM_ADDR dh, GM_ADDR doGrad, GM_ADDR dv,
                                      GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk,
                                      GM_ADDR dv2, GM_ADDR dg, GM_ADDR db, GM_ADDR dA, GM_ADDR userWS,
                                      const ChunkKdaBwdWyDqkgFusedTilingData &tiling)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(vNew));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(beta));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h));
        dhIn_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dh));
        do_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(doGrad));
        dvIn_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv));
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk));
        dv2_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv2));
        dg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dg));
        db_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(db));
        dAOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));

        // Per-core workspace: [core0 slots | core1 slots | ...]. BindCoreWorkspace()
        // rebinds wsF32_/wsT_ after coreIdx_ is known (Cube/Vec Init).
        userWS_ = userWS;
        f32BytesPerCore_ = static_cast<uint64_t>(NUM_GM_SLOTS) * SlotLayoutF32::TOTAL * sizeof(float);
        tBytesPerCore_ = static_cast<uint64_t>(NUM_GM_SLOTS) * SlotLayoutT::TOTAL * sizeof(T);
        wsF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWS_));
        wsT_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(userWS_ + f32BytesPerCore_));

        batch_ = static_cast<uint64_t>(tiling.batch);
        t_ = static_cast<uint64_t>(tiling.t);
        h_dim_ = static_cast<uint64_t>(tiling.h);
        hv_ = static_cast<uint64_t>(tiling.hv);
        kDim_ = static_cast<uint64_t>(tiling.k);
        vDim_ = static_cast<uint64_t>(tiling.v);
        bt_ = static_cast<uint64_t>(tiling.chunkSize);
        numChunks_ = static_cast<uint64_t>(tiling.numChunks);
        totalTasks_ = static_cast<uint64_t>(tiling.totalTasks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        hasVarlen_ = tiling.hasCuSeqlens != 0;
        scale_ = tiling.scale;
        stateVFirst_ = tiling.stateVFirst != 0;
        group_ = (h_dim_ == 0) ? 1 : (hv_ / h_dim_);
    }

    // Offset workspace to this AIC/AIV group's private slot bank.
    __aicore__ inline void BindCoreWorkspace(uint64_t coreIdx)
    {
        coreIdx_ = coreIdx;
        const uint64_t bytesPerCore = f32BytesPerCore_ + tBytesPerCore_;
        GM_ADDR coreWS = userWS_ + coreIdx_ * bytesPerCore;
        wsF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(coreWS));
        wsT_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(coreWS + f32BytesPerCore_));
    }

    __aicore__ inline bool ValidShapes() const
    {
        return !(usedCoreNum_ == 0 || totalTasks_ == 0 || kDim_ == 0 || vDim_ == 0 || bt_ == 0 || bt_ > MAX_BT ||
                 kDim_ > MAX_K_TOTAL || vDim_ > MAX_V_TOTAL || h_dim_ == 0 || hv_ == 0 || hv_ < h_dim_ ||
                 (hv_ % h_dim_) != 0);
    }

protected:
    __aicore__ inline uint32_t NumBk() const
    {
        return static_cast<uint32_t>((kDim_ + MAX_BK - 1) / MAX_BK);
    }
    __aicore__ inline uint32_t NumBv() const
    {
        return static_cast<uint32_t>((vDim_ + MAX_BV - 1) / MAX_BV);
    }
    __aicore__ inline uint32_t BkSize(uint32_t iK) const
    {
        const uint32_t remain = static_cast<uint32_t>(kDim_) - iK * MAX_BK;
        return remain < MAX_BK ? remain : MAX_BK;
    }
    __aicore__ inline uint32_t BvSize(uint32_t iV) const
    {
        const uint32_t remain = static_cast<uint32_t>(vDim_) - iV * MAX_BV;
        return remain < MAX_BV ? remain : MAX_BV;
    }

    // ---- GM tensor row offsets (BNSD) ----
    __aicore__ inline uint64_t QkOff(uint64_t b, uint64_t hh, uint64_t tok, uint64_t kOff = 0) const
    {
        return (((b * h_dim_ + hh) * t_ + tok) * kDim_) + kOff;
    }
    __aicore__ inline uint64_t HvKOff(uint64_t b, uint64_t hv, uint64_t tok, uint64_t kOff = 0) const
    {
        return (((b * hv_ + hv) * t_ + tok) * kDim_) + kOff;
    }
    __aicore__ inline uint64_t HvVOff(uint64_t b, uint64_t hv, uint64_t tok, uint64_t vOff = 0) const
    {
        return (((b * hv_ + hv) * t_ + tok) * vDim_) + vOff;
    }
    __aicore__ inline uint64_t BetaOff(uint64_t b, uint64_t hv, uint64_t tok) const
    {
        return (b * hv_ + hv) * t_ + tok;
    }
    __aicore__ inline uint64_t AOff(uint64_t b, uint64_t hv, uint64_t tok, uint64_t col = 0) const
    {
        return (((b * hv_ + hv) * t_ + tok) * bt_) + col;
    }
    // h/dh: [B,HV,numChunks,K,V] (stateVFirst=false) or [B,HV,numChunks,V,K] (stateVFirst=true).
    __aicore__ inline uint64_t StateOff(uint64_t b, uint64_t hv, uint64_t chunk) const
    {
        return ((b * hv_ + hv) * numChunks_ + chunk) * kDim_ * vDim_;
    }

    // ---- Workspace slot addressing ----
    __aicore__ inline uint64_t SlotBaseF32(uint64_t slot) const
    {
        return slot * SlotLayoutF32::TOTAL;
    }
    __aicore__ inline uint64_t SlotBaseT(uint64_t slot) const
    {
        return slot * SlotLayoutT::TOTAL;
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset,
                                        uint64_t count)
    {
        const uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && (rowBytes % 32) == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                         uint64_t count)
    {
        const uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && (rowBytes % 32) == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPad(dst[offset], src, params);
    }

    // task = iChunk * batch + iB (HV serial in-core).
    __aicore__ inline void DecodeChunkTask(uint64_t task, uint64_t &iB, uint64_t &iChunk) const
    {
        if (batch_ == 0) {
            iB = 0;
            iChunk = task;
            return;
        }
        iB = task % batch_;
        iChunk = task / batch_;
    }

    __aicore__ inline void ResolveChunkScalar(uint64_t iChunk, uint64_t iB, uint64_t &bos, uint64_t &localT,
                                              uint64_t &localChunk, uint64_t &bIdx) const
    {
        bos = 0;
        localT = t_;
        localChunk = iChunk;
        bIdx = iB;
        if (hasVarlen_) {
            const uint64_t seqId = static_cast<uint64_t>(chunkIndices_.GetValue(iChunk * 2));
            localChunk = static_cast<uint64_t>(chunkIndices_.GetValue(iChunk * 2 + 1));
            bos = static_cast<uint64_t>(cuSeqlens_.GetValue(seqId));
            const uint64_t eos = static_cast<uint64_t>(cuSeqlens_.GetValue(seqId + 1));
            localT = eos - bos;
            bIdx = 0;
        }
    }

    GlobalTensor<T> q_, k_, v_, vNew_, g_, beta_, a_, h_, dhIn_, do_, dvIn_, dv2_, wsT_;
    GlobalTensor<float> dq_, dk_, dg_, db_, dAOut_, wsF32_;
    GlobalTensor<int64_t> cuSeqlens_, chunkIndices_;
    GM_ADDR userWS_ = nullptr;
    uint64_t f32BytesPerCore_ = 0;
    uint64_t tBytesPerCore_ = 0;

    uint64_t batch_ = 0, t_ = 0, h_dim_ = 0, hv_ = 0, kDim_ = 0, vDim_ = 0, bt_ = 0, numChunks_ = 0, totalTasks_ = 0,
             usedCoreNum_ = 0, coreIdx_ = 0, group_ = 1;
    bool hasVarlen_ = false;
    bool stateVFirst_ = false;
    float scale_ = 1.0f;
};

// ---------------------------------------------------------------------------
// Build a (possibly sub-tiled) tla GM block view. `ambientRows`/`ambientCols`
// describe the *real* matrix this tile is carved from (so row/col strides are
// correct even when only a sub-range of columns is touched); `tileRowOff` /
// `tileColOff` select the sub-tile's start within that ambient matrix.
// ---------------------------------------------------------------------------
template <typename Element, typename LayoutTag>
__aicore__ inline auto MakeGmBlock(GlobalTensor<Element> &gm, uint64_t baseOffset, uint64_t ambientRows,
                                   uint64_t ambientCols, uint64_t tileRowOff, uint64_t tileColOff, uint64_t tileRows,
                                   uint64_t tileCols)
{
    auto layout = tla::MakeLayout<Element, LayoutTag>(ambientRows, ambientCols);
    auto tensor = tla::MakeTensor(gm[baseOffset], layout, Catlass::Arch::PositionGM{});
    return GetTile(tensor, tla::MakeCoord(tileRowOff, tileColOff), tla::MakeShape(tileRows, tileCols));
}

// ---------------------------------------------------------------------------
// Generic single-tile direct GEMM: C[m,n] = A[m,k] @ B[k,n], all operands <=128
// in every dim (fits one L0 tile, no ping-pong). ElementC may differ from the
// fp32 accumulator (FixPipe performs the cast, e.g. fp32 L0C -> bf16/fp16 GM).
// blockA/blockB/blockC are tla GM tile views built via MakeGmBlock (so callers
// control ambient strides for sub-tiled reads/writes). Modeled after
// chunk_bwd_dqkwg's TileGemmDirect / chunk_kda_fwd_intra_sub_chunk's
// ComputeScoreTile (single-buffer path).
// ---------------------------------------------------------------------------
// DirectTileGemm pipeline state (Cube-only).
// USE_FIX_MTE2_OVERLAP: after doFix, Set(FIX_MTE2) without Wait; next gemm (any
// doFix) must Wait before touching L0C/L1 — including L0C-accum tiles (doFix=false).
struct DirectTileGemmPipeState {
    bool fixMte2Outstanding = false;
    uint32_t l0Ping = 0;
};

template <typename ElementA, typename LayoutTagA, typename ElementB, typename LayoutTagB, typename ElementC,
         typename BlockA, typename BlockB, typename BlockC>
__aicore__ inline void DirectTileGemm(Catlass::Arch::Resource<KdaArchTag> &resource, BlockA &blockA, BlockB &blockB,
                                      BlockC &blockC, uint32_t m, uint32_t n, uint32_t k,
                                      DirectTileGemmPipeState *pipeState = nullptr, bool skipLoadA = false,
                                      bool initC = true, bool doFix = true)
{
    using LayoutTagC = Catlass::layout::RowMajor;
    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC,
                                               LayoutTagC>;

    const uint32_t l1ABytes = m * k * sizeof(ElementA);
    LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(0);
    LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(l1ABytes);
#if USE_L0_AB_DBUF
    const uint32_t l0Bytes = m * k * sizeof(ElementA);
    const uint32_t l0Ping = (pipeState != nullptr) ? (pipeState->l0Ping & 1U) : 0U;
    LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(l0Ping * l0Bytes);
    LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(l0Ping * (k * n * sizeof(ElementB)));
    if (pipeState != nullptr) {
        pipeState->l0Ping ^= 1U;
    }
#else
    LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
    LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
#endif
    LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<BlockA>;
    using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<BlockB>;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<BlockC>;
#else
    using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<BlockC>;
#endif
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

    auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
    auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
    auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
    auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
    auto layoutL0C = tla::MakeLayoutL0C(m, n);

    auto tL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
    auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
    auto tL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
    auto tL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
    auto tL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
    auto tileL1A = GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
    auto tileL1B = GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
    auto tileL0A = GetTile(tL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
    auto tileL0B = GetTile(tL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
    auto tileL0C = GetTile(tL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    TileMmad tileMmad;

    // HardEvent id: avoid Catlass-reserved CrossCore 8/9/10 collision surface.
    constexpr uint16_t evt = 14;
    // MTE2 may overlap prior Fix (different buffers). Drain Fix before L0C reuse.
    copyGmToL1B(tileL1B, blockB);
    SetFlag<HardEvent::MTE2_MTE1>(evt);
    WaitFlag<HardEvent::MTE2_MTE1>(evt);
    if (!skipLoadA) {
        copyGmToL1A(tileL1A, blockA);
        SetFlag<HardEvent::MTE2_MTE1>(evt);
        WaitFlag<HardEvent::MTE2_MTE1>(evt);
    }
#if USE_FIX_MTE2_OVERLAP
    if (pipeState != nullptr && pipeState->fixMte2Outstanding) {
        WaitFlag<HardEvent::FIX_MTE2>(evt);
        pipeState->fixMte2Outstanding = false;
    }
#endif
    copyL1ToL0B(tileL0B, tileL1B);
    copyL1ToL0A(tileL0A, tileL1A);
    SetFlag<HardEvent::MTE1_M>(evt);
    WaitFlag<HardEvent::MTE1_M>(evt);
    tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, initC, 0);
    if (doFix) {
        SetFlag<HardEvent::M_FIX>(evt);
        WaitFlag<HardEvent::M_FIX>(evt);
        SetFlag<HardEvent::M_MTE1>(evt);
        WaitFlag<HardEvent::M_MTE1>(evt);
        copyL0CToGm(blockC, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(evt);
#if USE_FIX_MTE2_OVERLAP
        if (pipeState != nullptr) {
            pipeState->fixMte2Outstanding = true; // next gemm / Process end drains
        } else {
            WaitFlag<HardEvent::FIX_MTE2>(evt);
        }
#else
        WaitFlag<HardEvent::FIX_MTE2>(evt);
#endif
    } else {
        // Keep L0C accum; free L0A/B for next tile load.
        SetFlag<HardEvent::M_MTE1>(evt);
        WaitFlag<HardEvent::M_MTE1>(evt);
    }
}

} // namespace kda_wy_dqkg

#endif // CHUNK_KDA_BWD_WY_DQKG_FUSED_COMMON_H
