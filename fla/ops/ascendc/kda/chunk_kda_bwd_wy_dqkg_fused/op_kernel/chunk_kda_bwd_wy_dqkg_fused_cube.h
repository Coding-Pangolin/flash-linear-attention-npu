/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaBwdWyDqkgFused — Cube (AIC) side.
 *
 * 2-head window (slot = winSlot^h, winSlot=(windowIdx&1)*2), stage-grouped:
 *   Stage0 WyV    : per head in window — dASlot/dvb —> Set(FLAG_C_S0)
 *   Stage1 KvAcc  : per BK, all heads — dq/dk/dw —> Set(FLAG_C_S1); Vec kg runs ∥ (no wait)
 *   Stage2 GateWy : per BK, all heads — Wait(V_GATE); dADelta/dkgb —> Set(FLAG_C_S2)
 *   Stage3 DaFinal: per head — Wait(V_MASK); dA@A;A@dA —> Set(FLAG_C_S3)
 *
 * I5 USE_WIN_SOFT_LEAD: Prefill Stage0 for KDA_BWD_PREFILL_WINDOWS; steady
 * RunWindowPost(w) then WaitFree(bank w)+Stage0(w+prefill).
 *
 * All GEMMs are single-tile direct (no ping-pong / dbuf) — see DirectTileGemm in
 * common.h. V-tile (bv) and K-tile (bk) accumulation across tiles is done by the
 * Vector side summing per-tile "delta" slots (dASlot/dqSlot/dkSlot/dwSlot), so no
 * extra CrossCore round trips are needed for V>128.
 *
 * NOTE: dwNegWs / kgWs rows >= validRows are zero-padded by the Vector gate step
 * (chunk_kda_bwd_wy_dqkg_fused_vector.h) so Stage2.2's full-BT contraction over
 * those workspace planes is safe without per-token OOB reads.
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_CUBE_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_CUBE_H

#include "chunk_kda_bwd_wy_dqkg_fused_common.h"
#include <type_traits>

namespace kda_wy_dqkg {

template <typename T>
class KdaWyDqkgCube : public KdaWyDqkgBase<T> {
    using Base = KdaWyDqkgBase<T>;
    using Base::bt_;
    using Base::kDim_;
    using Base::vDim_;
    using Base::hv_;
    using Base::totalTasks_;
    using Base::usedCoreNum_;
    using Base::coreIdx_;
    using Base::stateVFirst_;
    using Base::wsF32_;
    using Base::wsT_;
    using Base::a_;
    using Base::dvIn_;
    using Base::v_;
    using Base::do_;
    using Base::vNew_;
    using Base::h_;
    using Base::dhIn_;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR g, GM_ADDR beta, GM_ADDR a,
                                GM_ADDR h, GM_ADDR dh, GM_ADDR doGrad, GM_ADDR dv, GM_ADDR cuSeqlens,
                                GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv2, GM_ADDR dg, GM_ADDR db,
                                GM_ADDR dA, GM_ADDR userWS, const ChunkKdaBwdWyDqkgFusedTilingData &tiling,
                                TPipe *pipe)
    {
        (void)pipe;
        this->InitCommon(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db,
                         dA, userWS, tiling);
        this->BindCoreWorkspace(static_cast<uint64_t>(GetBlockIdx()));
    }

    __aicore__ inline void Process()
    {
        if (!this->ValidShapes()) {
            return;
        }
        Catlass::Arch::Resource<KdaArchTag> resource;
        DirectTileGemmPipeState gemmPipe_{};
        const uint32_t nBv = this->NumBv();
        const uint32_t nBk = this->NumBk();
        uint64_t tasksOnCore = 0;

        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);
            ++tasksOnCore;

            const uint64_t chunkStart = localChunk * bt_;
            if (chunkStart >= localT) {
                continue;
            }
            const uint64_t validRows = (chunkStart + bt_ <= localT) ? bt_ : (localT - chunkStart);
            const uint64_t tok0 = bos + chunkStart;
            const uint64_t nHvWin = (hv_ + 1ULL) / 2ULL;
            if (nHvWin == 0) {
                continue;
            }

#if USE_WIN_SOFT_LEAD
            // I5 (isub VEC_2WIN): Prefill Stage0×prefill; steady Post(w) then Stage0(w+2).
            // WaitFree only before Stage0 on a reused bank (Store of that bank done).
            constexpr uint64_t kPrefillCap = static_cast<uint64_t>(KDA_BWD_PREFILL_WINDOWS);
            const uint64_t prefill = (nHvWin < kPrefillCap) ? nHvWin : kPrefillCap;
            for (uint64_t w = 0; w < prefill; ++w) {
                RunWindowStage0(resource, gemmPipe_, w, bIdx, tok0, validRows, nBv);
            }
            for (uint64_t w = 0; w < nHvWin; ++w) {
                RunWindowPost(resource, gemmPipe_, w, bIdx, tok0, localChunk, validRows, nBv, nBk);
                if (w + prefill < nHvWin) {
                    const uint64_t wn = w + prefill;
                    const uint32_t winSlot = static_cast<uint32_t>((w & 1ULL) * 2ULL);
                    const uint64_t hvBase = w * 2ULL;
                    const uint32_t headCnt =
                        (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
                    for (uint32_t h = 0; h < headCnt; ++h) {
                        Catlass::Arch::CrossCoreWaitFlag(slotFree_[winSlot ^ h]);
                    }
                    RunWindowStage0(resource, gemmPipe_, wn, bIdx, tok0, validRows, nBv);
                }
            }
#else
            uint64_t windowIdx = 0;
            for (uint64_t hvBase = 0; hvBase < hv_; hvBase += 2) {
                const uint32_t headCnt =
                    (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
                const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);

                if (windowIdx >= 2) {
                    for (uint32_t h = 0; h < headCnt; ++h) {
                        Catlass::Arch::CrossCoreWaitFlag(slotFree_[winSlot ^ h]);
                    }
                }

                RunWindowStage0(resource, gemmPipe_, windowIdx, bIdx, tok0, validRows, nBv);
                RunWindowPost(resource, gemmPipe_, windowIdx, bIdx, tok0, localChunk, validRows, nBv, nBk);
                ++windowIdx;
            }
#endif
        }
        if (tasksOnCore > 0) {
            for (uint32_t s = 0; s < NUM_GM_SLOTS; ++s) {
                Catlass::Arch::CrossCoreWaitFlag(slotFree_[s]);
            }
        }
#if USE_FIX_MTE2_OVERLAP
        if (gemmPipe_.fixMte2Primed) {
            WaitFlag<HardEvent::FIX_MTE2>(9);
        }
#endif
    }

private:
    __aicore__ inline void RunWindowStage0(Catlass::Arch::Resource<KdaArchTag> &resource,
                                           DirectTileGemmPipeState &pipe, uint64_t windowIdx, uint64_t bIdx,
                                           uint64_t tok0, uint64_t validRows, uint32_t nBv)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
        const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = winSlot ^ h;
            const uint64_t iHv = hvBase + h;
            RunStage0(resource, pipe, slot, bIdx, iHv, tok0, validRows, nBv);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cS0_);
        }
    }

    __aicore__ inline void RunWindowPost(Catlass::Arch::Resource<KdaArchTag> &resource,
                                         DirectTileGemmPipeState &pipe, uint64_t windowIdx, uint64_t bIdx,
                                         uint64_t tok0, uint64_t localChunk, uint64_t validRows, uint32_t nBv,
                                         uint32_t nBk)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
        const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);

        // Wait Stage0Vec (dv2/db/dAWs) before any Stage1 — once per head / window.
        for (uint32_t h = 0; h < headCnt; ++h) {
            Catlass::Arch::CrossCoreWaitFlag(vS0_);
        }

        for (uint32_t iK = 0; iK < nBk; ++iK) {
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
                RunStage1(resource, pipe, slot, bIdx, iHv, tok0, localChunk, validRows, nBv, iK);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cS1_);
            }
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
#if USE_STAGE2_PRELOAD_A
                PreloadAToL1(resource, bIdx, iHv, tok0, validRows);
#endif
                Catlass::Arch::CrossCoreWaitFlag(vGate_);
                RunStage2(resource, pipe, slot, bIdx, iHv, tok0, validRows, iK);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cS2_);
            }
#if USE_EARLY_MASK_PER_HEAD
            if (iK + 1 == nBk) {
                for (uint32_t h = 0; h < headCnt; ++h) {
                    const uint64_t slot = winSlot ^ h;
                    const uint64_t iHv = hvBase + h;
                    Catlass::Arch::CrossCoreWaitFlag(vMask_);
                    RunStage3(resource, pipe, slot, bIdx, iHv, tok0, validRows);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cS3_);
                }
            }
#endif
        }

#if !USE_EARLY_MASK_PER_HEAD
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = winSlot ^ h;
            const uint64_t iHv = hvBase + h;
            Catlass::Arch::CrossCoreWaitFlag(vMask_);
            RunStage3(resource, pipe, slot, bIdx, iHv, tok0, validRows);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cS3_);
        }
#endif
    }

    // ---- Stage0: dASlot[iv] = dv_iv @ v_iv^T ; dvbWs[:,iv] = A @ dv_iv ----
    __aicore__ inline void RunStage0(Catlass::Arch::Resource<KdaArchTag> &resource, DirectTileGemmPipeState &pipe,
                                     uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t validRows,
                                     uint32_t nBv)
    {
        const uint64_t f32Base = this->SlotBaseF32(slot);
        const uint64_t dvBase = this->HvVOff(bIdx, iHv, 0, 0);
        const uint64_t vBase = this->HvVOff(bIdx, iHv, 0, 0);
        const uint64_t aBase = this->AOff(bIdx, iHv, 0, 0);

        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;

            auto blockDv = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
            auto blockVT = MakeGmBlock<T, ColumnMajor>(v_, vBase, vDim_, this->t_, vOff, tok0, bv, validRows);
#if USE_STAGE0_DA_L0C_ACCUM
            // Accum across V-tiles → one Fix into dAWs (Vec skips Σ dASlot).
            const bool initC = (iv == 0);
            const bool doFix = (iv + 1 == nBv);
            auto blockDaAcc = MakeGmBlock<float, RowMajor>(wsF32_, f32Base + SlotLayoutF32::dAWs, MAX_BT, MAX_BT, 0,
                                                           0, validRows, validRows);
            DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(
                resource, blockDv, blockVT, blockDaAcc, static_cast<uint32_t>(validRows),
                static_cast<uint32_t>(validRows), bv, &pipe, false, initC, doFix);
#else
            auto blockDaSlot = MakeGmBlock<float, RowMajor>(
                wsF32_, f32Base + SlotLayoutF32::dASlot + static_cast<uint64_t>(iv) * MAX_BT * MAX_BT, MAX_BT,
                MAX_BT, 0, 0, validRows, validRows);
            DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDv, blockVT, blockDaSlot,
                                                               static_cast<uint32_t>(validRows),
                                                               static_cast<uint32_t>(validRows), bv, &pipe);
#endif
        }

#if USE_L1_A_RESIDENT
        // Load A once, reuse across V-tiles for A @ dv (full BT only; partial → Vec).
        if (validRows >= bt_) {
            auto blockA = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, validRows, validRows);
            auto blockDv0 = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, 0, validRows,
                                                     this->BvSize(0));
            auto blockDvb0 = MakeGmBlock<float, RowMajor>(wsF32_, f32Base + SlotLayoutF32::dvbWs, MAX_BT, MAX_BV, 0,
                                                          0, validRows, this->BvSize(0));
            DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockA, blockDv0, blockDvb0,
                                                            static_cast<uint32_t>(validRows), this->BvSize(0),
                                                            static_cast<uint32_t>(validRows), &pipe, false);
            for (uint32_t iv = 1; iv < nBv; ++iv) {
                const uint32_t bv = this->BvSize(iv);
                const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
                const uint64_t panel = f32Base + SlotLayoutF32::dvbWs + static_cast<uint64_t>(iv) * MAX_BT * MAX_BV;
                auto blockASkip = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, validRows, validRows);
                auto blockDvB = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDvbWs =
                    MakeGmBlock<float, RowMajor>(wsF32_, panel, MAX_BT, MAX_BV, 0, 0, validRows, bv);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockASkip, blockDvB, blockDvbWs,
                                                                static_cast<uint32_t>(validRows), bv,
                                                                static_cast<uint32_t>(validRows), &pipe, true);
            }
        }
#else
        // Full BT: Cube A@dv into contiguous panels. Partial BT: Vec recomputes A@dv
        // (2nd V-tile FixPipe into panel was observed all-zero when validRows=32).
        if (validRows >= bt_) {
            for (uint32_t iv = 0; iv < nBv; ++iv) {
                const uint32_t bv = this->BvSize(iv);
                const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
                const uint64_t panel = f32Base + SlotLayoutF32::dvbWs + static_cast<uint64_t>(iv) * MAX_BT * MAX_BV;
                auto blockA = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, validRows, validRows);
                auto blockDvB = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDvbWs = MakeGmBlock<float, RowMajor>(wsF32_, panel, MAX_BT, MAX_BV, 0, 0, validRows, bv);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockA, blockDvB, blockDvbWs,
                                                                static_cast<uint32_t>(validRows), bv,
                                                                static_cast<uint32_t>(validRows), &pipe);
            }
        }
#endif
    }

    // ---- Stage1: dq/dk/dw over V-tiles for this BK ----
    // Default: Fix each iv → dqSlot[iv]; Gate sums planes.
    // USE_STAGE1_L0C_ACCUM: L0C accum across iv, one Fix → plane 0; Gate single load.
    __aicore__ inline void RunStage1(Catlass::Arch::Resource<KdaArchTag> &resource, DirectTileGemmPipeState &pipe,
                                     uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t localChunk,
                                     uint64_t validRows, uint32_t nBv, uint32_t iK)
    {
        if (stateVFirst_) {
            RunStage1Impl<true>(resource, pipe, slot, bIdx, iHv, tok0, localChunk, validRows, nBv, iK);
        } else {
            RunStage1Impl<false>(resource, pipe, slot, bIdx, iHv, tok0, localChunk, validRows, nBv, iK);
        }
    }

    template <bool StateVFirst>
    __aicore__ inline void RunStage1Impl(Catlass::Arch::Resource<KdaArchTag> &resource, DirectTileGemmPipeState &pipe,
                                         uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0,
                                         uint64_t localChunk, uint64_t validRows, uint32_t nBv, uint32_t iK)
    {
        const uint64_t f32Base = this->SlotBaseF32(slot);
        const uint32_t bk = this->BkSize(iK);
        const uint64_t kOff = static_cast<uint64_t>(iK) * MAX_BK;
        const uint64_t doBase = this->HvVOff(bIdx, iHv, 0, 0);
        const uint64_t vNewBase = this->HvVOff(bIdx, iHv, 0, 0);
        const uint64_t dvBase = this->HvVOff(bIdx, iHv, 0, 0);
        const uint64_t stateBase = this->StateOff(bIdx, iHv, localChunk);
        const uint32_t m = static_cast<uint32_t>(validRows);

#if USE_STAGE1_L0C_ACCUM
        // One product at a time (shared L0C): dq then dk then dw; Fix only on last iv → plane 0.
        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
            const bool initC = (iv == 0);
            const bool doFix = (iv + 1 == nBv);
            auto blockDo = MakeGmBlock<T, RowMajor>(do_, doBase, this->t_, vDim_, tok0, vOff, validRows, bv);
            auto blockDqSlot = MakeGmBlock<float, RowMajor>(
                wsF32_, f32Base + SlotLayoutF32::dqSlot, MAX_BT, MAX_BK, 0, 0, validRows, bk);
            if constexpr (StateVFirst) {
                auto blockH = MakeGmBlock<T, RowMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockDo, blockH, blockDqSlot, m, bk, bv,
                                                                &pipe, false, initC, doFix);
            } else {
                auto blockH = MakeGmBlock<T, ColumnMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDo, blockH, blockDqSlot, m, bk, bv,
                                                                   &pipe, false, initC, doFix);
            }
        }
        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
            const bool initC = (iv == 0);
            const bool doFix = (iv + 1 == nBv);
            auto blockVNew =
                MakeGmBlock<T, RowMajor>(vNew_, vNewBase, this->t_, vDim_, tok0, vOff, validRows, bv);
            auto blockDkSlot = MakeGmBlock<float, RowMajor>(
                wsF32_, f32Base + SlotLayoutF32::dkSlot, MAX_BT, MAX_BK, 0, 0, validRows, bk);
            if constexpr (StateVFirst) {
                auto blockDh = MakeGmBlock<T, RowMajor>(dhIn_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockVNew, blockDh, blockDkSlot, m, bk, bv,
                                                                &pipe, false, initC, doFix);
            } else {
                auto blockDh = MakeGmBlock<T, ColumnMajor>(dhIn_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockVNew, blockDh, blockDkSlot, m, bk,
                                                                   bv, &pipe, false, initC, doFix);
            }
        }
        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
            const bool initC = (iv == 0);
            const bool doFix = (iv + 1 == nBv);
            auto blockDv = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
            auto blockDwSlot = MakeGmBlock<float, RowMajor>(
                wsF32_, f32Base + SlotLayoutF32::dwSlot, MAX_BT, MAX_BK, 0, 0, validRows, bk);
            if constexpr (StateVFirst) {
                auto blockH = MakeGmBlock<T, RowMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockDv, blockH, blockDwSlot, m, bk, bv,
                                                                &pipe, false, initC, doFix);
            } else {
                auto blockH = MakeGmBlock<T, ColumnMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDv, blockH, blockDwSlot, m, bk, bv,
                                                                   &pipe, false, initC, doFix);
            }
        }
#else
        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
            const uint64_t slotElemOff = static_cast<uint64_t>(iv) * MAX_BT * MAX_BK;

            // stateVFirst: physical [V,K] is already RowMajor B[v,k]=h[v,k].
            // !stateVFirst: physical [K,V] viewed as ColumnMajor(V,K) ⇒ B[v,k]=h[k,v].
            if constexpr (StateVFirst) {
                auto blockH = MakeGmBlock<T, RowMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                auto blockDh = MakeGmBlock<T, RowMajor>(dhIn_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);

                auto blockDo = MakeGmBlock<T, RowMajor>(do_, doBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDqSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dqSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockDo, blockH, blockDqSlot, m, bk, bv,
                                                                &pipe);

                auto blockVNew =
                    MakeGmBlock<T, RowMajor>(vNew_, vNewBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDkSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dkSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockVNew, blockDh, blockDkSlot, m, bk, bv,
                                                                &pipe);

                auto blockDv = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDwSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dwSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockDv, blockH, blockDwSlot, m, bk, bv,
                                                                &pipe);
            } else {
                auto blockH = MakeGmBlock<T, ColumnMajor>(h_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);
                auto blockDh = MakeGmBlock<T, ColumnMajor>(dhIn_, stateBase, vDim_, kDim_, vOff, kOff, bv, bk);

                auto blockDo = MakeGmBlock<T, RowMajor>(do_, doBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDqSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dqSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDo, blockH, blockDqSlot, m, bk, bv,
                                                                   &pipe);

                auto blockVNew =
                    MakeGmBlock<T, RowMajor>(vNew_, vNewBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDkSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dkSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockVNew, blockDh, blockDkSlot, m, bk,
                                                                   bv, &pipe);

                auto blockDv = MakeGmBlock<T, RowMajor>(dvIn_, dvBase, this->t_, vDim_, tok0, vOff, validRows, bv);
                auto blockDwSlot = MakeGmBlock<float, RowMajor>(
                    wsF32_, f32Base + SlotLayoutF32::dwSlot + slotElemOff, MAX_BT, MAX_BK, 0, 0, validRows, bk);
                DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDv, blockH, blockDwSlot, m, bk, bv,
                                                                   &pipe);
            }
        }
#endif
    }

    // ---- Stage2: dADeltaWs = dwNegWs @ kgWs^T ; dkgbWs = A @ dwNegWs ----
#if USE_STAGE2_PRELOAD_A
    __aicore__ inline void PreloadAToL1(Catlass::Arch::Resource<KdaArchTag> &resource, uint64_t bIdx, uint64_t iHv,
                                        uint64_t tok0, uint64_t validRows)
    {
        // Mirror DirectTileGemm L1A placement so Stage2 A@dwNeg can skipLoadA.
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy =
            Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, T, LayoutTagA, T, LayoutTagA, float, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;

        const uint32_t m = static_cast<uint32_t>(validRows);
        const uint32_t k = static_cast<uint32_t>(bt_);
        const uint64_t aBase = this->AOff(bIdx, iHv, 0, 0);
        auto blockA = MakeGmBlock<T, LayoutTagA>(a_, aBase, this->t_, bt_, tok0, 0, validRows, bt_);
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;

        LocalTensor<T> l1A = resource.l1Buf.template GetBufferByByte<T>(0);
        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(m, k);
        auto tL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tileL1A = GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        CopyGmToL1A copyGmToL1A;
        constexpr uint16_t evt = 14;
        copyGmToL1A(tileL1A, blockA);
        SetFlag<HardEvent::MTE2_MTE1>(evt);
        WaitFlag<HardEvent::MTE2_MTE1>(evt);
    }
#endif

    __aicore__ inline void RunStage2(Catlass::Arch::Resource<KdaArchTag> &resource, DirectTileGemmPipeState &pipe,
                                     uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t validRows,
                                     uint32_t iK)
    {
        const uint64_t f32Base = this->SlotBaseF32(slot);
        const uint64_t tBase = this->SlotBaseT(slot);
        const uint32_t bk = this->BkSize(iK);
        const uint64_t aBase = this->AOff(bIdx, iHv, 0, 0);

        auto blockDwNeg =
            MakeGmBlock<T, RowMajor>(wsT_, tBase + SlotLayoutT::dwNegWs, MAX_BT, MAX_BK, 0, 0, bt_, bk);
        auto blockA = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, validRows, bt_);
        auto blockDkgb =
            MakeGmBlock<float, RowMajor>(wsF32_, f32Base + SlotLayoutF32::dkgbWs, MAX_BT, MAX_BK, 0, 0, validRows,
                                         bk);
        auto blockKgT =
            MakeGmBlock<T, ColumnMajor>(wsT_, tBase + SlotLayoutT::kgWs, MAX_BK, MAX_BT, 0, 0, bk, bt_);
        auto blockDaDelta =
            MakeGmBlock<float, RowMajor>(wsF32_, f32Base + SlotLayoutF32::dADeltaWs, MAX_BT, MAX_BT, 0, 0, bt_, bt_);

#if USE_STAGE2_PRELOAD_A
        // A already in L1 from PreloadAToL1 — do A@dwNeg first (skipLoadA), then dwNeg@kg.
        DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockA, blockDwNeg, blockDkgb,
                                                        static_cast<uint32_t>(validRows), bk,
                                                        static_cast<uint32_t>(bt_), &pipe, /*skipLoadA=*/true);
        DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDwNeg, blockKgT, blockDaDelta,
                                                           static_cast<uint32_t>(bt_), static_cast<uint32_t>(bt_),
                                                           bk, &pipe);
#else
        DirectTileGemm<T, RowMajor, T, ColumnMajor, float>(resource, blockDwNeg, blockKgT, blockDaDelta,
                                                           static_cast<uint32_t>(bt_), static_cast<uint32_t>(bt_),
                                                           bk, &pipe);
        DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockA, blockDwNeg, blockDkgb,
                                                        static_cast<uint32_t>(validRows), bk,
                                                        static_cast<uint32_t>(bt_), &pipe);
#endif
    }

    // ---- Stage3: dA2InterimWs = dAMaskedWs @ A (cast->T) ; dA3Ws = A @ dA2InterimWs ----
    __aicore__ inline void RunStage3(Catlass::Arch::Resource<KdaArchTag> &resource, DirectTileGemmPipeState &pipe,
                                     uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t validRows)
    {
        const uint64_t f32Base = this->SlotBaseF32(slot);
        const uint64_t tBase = this->SlotBaseT(slot);
        const uint64_t aBase = this->AOff(bIdx, iHv, 0, 0);
        // Use validRows×validRows (golden) — full BT×BT on partial chunks reads OOB A / garbage.
        const uint32_t vr = static_cast<uint32_t>(validRows);

        auto blockMasked =
            MakeGmBlock<T, RowMajor>(wsT_, tBase + SlotLayoutT::dAMaskedWs, MAX_BT, MAX_BT, 0, 0, vr, vr);
        auto blockA1 = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, vr, vr);
        auto blockInterim =
            MakeGmBlock<T, RowMajor>(wsT_, tBase + SlotLayoutT::dA2InterimWs, MAX_BT, MAX_BT, 0, 0, vr, vr);
        DirectTileGemm<T, RowMajor, T, RowMajor, T>(resource, blockMasked, blockA1, blockInterim, vr, vr, vr, &pipe);

        auto blockA2 = MakeGmBlock<T, RowMajor>(a_, aBase, this->t_, bt_, tok0, 0, vr, vr);
        auto blockInterim2 =
            MakeGmBlock<T, RowMajor>(wsT_, tBase + SlotLayoutT::dA2InterimWs, MAX_BT, MAX_BT, 0, 0, vr, vr);
        auto blockDa3 =
            MakeGmBlock<float, RowMajor>(wsF32_, f32Base + SlotLayoutF32::dA3Ws, MAX_BT, MAX_BT, 0, 0, vr, vr);
        DirectTileGemm<T, RowMajor, T, RowMajor, float>(resource, blockA2, blockInterim2, blockDa3, vr, vr, vr,
                                                        &pipe);
    }

    Catlass::Arch::CrossCoreFlag cS0_{FLAG_C_S0};
    Catlass::Arch::CrossCoreFlag cS1_{FLAG_C_S1};
    Catlass::Arch::CrossCoreFlag vGate_{FLAG_V_GATE};
    Catlass::Arch::CrossCoreFlag cS2_{FLAG_C_S2};
    Catlass::Arch::CrossCoreFlag vMask_{FLAG_V_MASK};
    Catlass::Arch::CrossCoreFlag cS3_{FLAG_C_S3};
    Catlass::Arch::CrossCoreFlag vS0_{FLAG_V_S0};
    Catlass::Arch::CrossCoreFlag slotFree_[NUM_GM_SLOTS] = {FLAG_SLOT_FREE0, FLAG_SLOT_FREE1, FLAG_SLOT_FREE2,
                                                             FLAG_SLOT_FREE3};
};

} // namespace kda_wy_dqkg

#endif // CHUNK_KDA_BWD_WY_DQKG_FUSED_CUBE_H
