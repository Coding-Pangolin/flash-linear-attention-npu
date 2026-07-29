/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaBwdWyDqkgFused — Vector (AIV) side.
 *
 * 2-head stage-grouped window (mirrors Cube Process). Dual AIV splits row work via
 * GetSubBlockIdx(); SetV*Joined keeps Barrier+Set so Cube's single Wait credit matches.
 * I5 USE_WIN_SOFT_LEAD: Prefill Stage0Vec×prefill; steady Post then Stage0Vec(w+prefill).
 *
 *   Stage0Vec  : wait(FLAG_C_S0); dv2/db from dvbWs; dAWs = sum_v dASlot[v]
 *   KgVec      : (∥ Cube Stage1) k/g → gk/kg ws; park gn in dgkWs (no Wait C_S1)
 *   GateOnlyVec: wait(FLAG_C_S1); gate dq/dk/dw/dgk → ws; set(FLAG_V_GATE)
 *   EpilogVec  : wait(FLAG_C_S2); db/dg/dk/dq finalize + store; dAWs += dADelta
 *   Stage3Mask : mask dA → dAMaskedWs; set(FLAG_V_MASK)
 *   Stage3Store: wait(FLAG_C_S3); store dA/db; set(slotFree_[slot])
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_VECTOR_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_VECTOR_H

#include "chunk_kda_bwd_wy_dqkg_fused_common.h"

namespace kda_wy_dqkg {

// UB scratch: Stage0 and Gate/Epilog/Stage3 are sequential — share one F32 + one T arena.
// Peak = Gate/Epilog live panels. With USE_OWNED_ARENA: 8*(BT/2)*BK; else 8*BT*BK.
// AtlasA2 AIV UB ≈192KB: 128KB F32 + 48KB T + ~4KB small ≈180KB.
constexpr uint32_t ARENA_F32_ELEMS = 8 * ARENA_BT_ROWS * MAX_BK;
constexpr uint32_t ARENA_T_ELEMS = 6 * ARENA_BT_ROWS * MAX_BK;

constexpr uint16_t EVT_MTE2_V = 0;
constexpr uint16_t EVT_V_MTE2 = 1;
constexpr uint16_t EVT_V_MTE3 = 2;
constexpr uint16_t EVT_MTE3_V = 3;
constexpr uint16_t EVT_MTE3_MTE2 = 4;
constexpr uint16_t EVT_SCALAR_VS = 5;
constexpr uint16_t EVT_SCALAR_SV = 6;
constexpr uint16_t EVT_S_V2 = 7;
// Dedicated IDs for Gate state dgk row loop — must not share EVT_MTE2_V / SCALAR
// with SyncMte2ToV or BK0→BK1 leaves stale credits and BK1 loads race.
constexpr uint16_t EVT_STATE_MTE2_V = 8;
constexpr uint16_t EVT_STATE_VS = 9;
constexpr uint16_t EVT_STATE_SV = 10;

template <typename T>
class KdaWyDqkgVector : public KdaWyDqkgBase<T> {
    using Base = KdaWyDqkgBase<T>;
    using Base::bt_;
    using Base::kDim_;
    using Base::vDim_;
    using Base::hv_;
    using Base::group_;
    using Base::totalTasks_;
    using Base::usedCoreNum_;
    using Base::coreIdx_;
    using Base::scale_;
    using Base::stateVFirst_;
    using Base::stageId_;
    using Base::taskBegin_;
    using Base::taskEnd_;
    using Base::SlotOf;
    using Base::SetTaskBank;
    using Base::q_;
    using Base::k_;
    using Base::v_;
    using Base::vNew_;
    using Base::g_;
    using Base::beta_;
    using Base::a_;
    using Base::h_;
    using Base::dhIn_;
    using Base::do_;
    using Base::dvIn_;
    using Base::dv2_;
    using Base::dq_;
    using Base::dk_;
    using Base::dg_;
    using Base::db_;
    using Base::dAOut_;
    using Base::wsF32_;
    using Base::wsT_;

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR g, GM_ADDR beta, GM_ADDR a,
                                GM_ADDR h, GM_ADDR dh, GM_ADDR doGrad, GM_ADDR dv, GM_ADDR cuSeqlens,
                                GM_ADDR chunkIndices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv2, GM_ADDR dg, GM_ADDR db,
                                GM_ADDR dA, GM_ADDR userWS, const ChunkKdaBwdWyDqkgFusedTilingData &tiling,
                                TPipe *pipe)
    {
        this->InitCommon(q, k, v, vNew, g, beta, a, h, dh, doGrad, dv, cuSeqlens, chunkIndices, dq, dk, dv2, dg, db,
                         dA, userWS, tiling);
        pipe_ = pipe;
        subBlockNum_ = static_cast<uint32_t>(GetSubBlockNum());
        if (subBlockNum_ == 0) {
            subBlockNum_ = 1;
        }
        subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        this->BindCoreWorkspace(static_cast<uint64_t>(GetBlockIdx()) / static_cast<uint64_t>(subBlockNum_));

        pipe_->InitBuffer(arenaF32_, ARENA_F32_ELEMS * sizeof(float));
        pipe_->InitBuffer(arenaT_, ARENA_T_ELEMS * sizeof(T));
        pipe_->InitBuffer(betaBuf_, MAX_BT * sizeof(float));
        pipe_->InitBuffer(dbAccBuf_, MAX_BT * sizeof(float));
        pipe_->InitBuffer(smallBuf_, 4 * MAX_BK * sizeof(float));
        // Brcb / WholeReduceSum scratch: 8 floats per row.
        pipe_->InitBuffer(brcbBuf_, MAX_BT * 8 * sizeof(float));
        pipe_->InitBuffer(maskBuf_, MAX_BT * ((MAX_BT + 7) / 8) * sizeof(uint8_t));
#if USE_MASK_SELECT_SLIM
        // 8-float Select zero pattern — built once; Stage3 hot path skips Duplicate.
        pipe_->InitBuffer(zeroSelBuf_, 8 * sizeof(float));
#endif
        // P2a: prebuild mask for full-BT (model hot path); EnsureMask is then a no-op.
        {
            LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
            BuildStrictLowerMaskPacked(mask, static_cast<uint32_t>(bt_));
            SetFlag<HardEvent::S_V>(EVT_SCALAR_SV);
            WaitFlag<HardEvent::S_V>(EVT_SCALAR_SV);
            cachedMaskValidRows_ = static_cast<uint32_t>(bt_);
        }
#if USE_MASK_SELECT_SLIM
        {
            LocalTensor<float> zero = zeroSelBuf_.Get<float>();
            Duplicate(zero, 0.0f, 8);
            PipeBarrier<PIPE_V>();
        }
#endif
#if USE_VEC_MTE2_PP
        InitVecMte2Pp();
#endif
    }

    __aicore__ inline void Process()
    {
        if (!this->ValidShapes()) {
            return;
        }
        if (stageId_ == 1) {
            ProcessStageA();
            return;
        }
        if (stageId_ == 2) {
            ProcessStageB();
            return;
        }
        if (stageId_ == 3) {
            ProcessStageC();
            return;
        }
        ProcessFused();
    }

    __aicore__ inline void ProcessFused()
    {
        // Process bookend SetFree×4 (isub / I5). Hot-path depth uses C_S*/V_* +
        // WaitFree before Cube Stage0(w+2) on a reused bank.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
        for (uint32_t s = 0; s < NUM_GM_SLOTS; ++s) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(slotFree_[s]);
        }

        const uint32_t nBv = this->NumBv();
        const uint32_t nBk = this->NumBk();

        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);

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
            constexpr uint64_t kPrefillCap = static_cast<uint64_t>(KDA_BWD_PREFILL_WINDOWS);
            const uint64_t prefill = (nHvWin < kPrefillCap) ? nHvWin : kPrefillCap;
            for (uint64_t w = 0; w < prefill; ++w) {
                RunWindowStage0Vec(w, bIdx, tok0, validRows, nBv);
            }
#if USE_WIN_SOFT_LEAD_V2
            // Mirror Cube: PostBody(w) → Stage0Vec(w+1) → PostTail(w).
            // Stage0Vec(next) Wait cS0 while Cube may still be in Stage3(w).
            for (uint64_t w = 0; w < nHvWin; ++w) {
                RunWindowPostBodyVec(w, bIdx, tok0, localChunk, validRows, nBv, nBk);
                if (w + 1 < nHvWin) {
                    RunWindowStage0Vec(w + 1, bIdx, tok0, validRows, nBv);
                }
                RunWindowPostTailVec(w, bIdx, tok0, validRows);
            }
#else
            for (uint64_t w = 0; w < nHvWin; ++w) {
                RunWindowPostVec(w, bIdx, tok0, localChunk, validRows, nBv, nBk);
                if (w + prefill < nHvWin) {
                    RunWindowStage0Vec(w + prefill, bIdx, tok0, validRows, nBv);
                }
            }
#endif
#else
            for (uint64_t w = 0; w < nHvWin; ++w) {
                RunWindowStage0Vec(w, bIdx, tok0, validRows, nBv);
                RunWindowPostVec(w, bIdx, tok0, localChunk, validRows, nBv, nBk);
            }
#endif
        }
#if USE_VEC_MTE2_PP
        ReleaseVecMte2Pp();
#endif
    }

    // F6 OpA: Stage0Vec + Kg (∥ Cube Stage1). No Gate/Epilog/Mask.
    __aicore__ inline void ProcessStageA()
    {
        const uint32_t nBv = this->NumBv();
        const uint32_t nBk = this->NumBk();
        for (uint64_t task = taskBegin_ + coreIdx_; task < taskEnd_; task += usedCoreNum_) {
            this->SetTaskBank(task);
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);
            const uint64_t chunkStart = localChunk * bt_;
            if (chunkStart >= localT) {
                continue;
            }
            const uint64_t validRows = (chunkStart + bt_ <= localT) ? bt_ : (localT - chunkStart);
            const uint64_t tok0 = bos + chunkStart;
            const uint64_t nHvWin = (hv_ + 1ULL) / 2ULL;
            for (uint64_t w = 0; w < nHvWin; ++w) {
                RunWindowStage0VecStage(w, bIdx, tok0, validRows, nBv);
                for (uint32_t iK = 0; iK < nBk; ++iK) {
                    const uint64_t hvBase = w * 2ULL;
                    const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
                    for (uint32_t h = 0; h < headCnt; ++h) {
                        const uint64_t slot = this->SlotOf(w, h);
                        const uint64_t iHv = hvBase + h;
                        const uint64_t iH = iHv / group_;
                        KgVec(slot, bIdx, iHv, iH, tok0, validRows, iK);
                    }
                }
            }
        }
#if USE_VEC_MTE2_PP
        ReleaseVecMte2Pp();
#endif
    }

    // F6 OpB: Gate + Epilog. No Wait C_S1 (OpA GM visible via stream order). No Mask.
    __aicore__ inline void ProcessStageB()
    {
        const uint32_t nBv = this->NumBv();
        const uint32_t nBk = this->NumBk();
        for (uint64_t task = taskBegin_ + coreIdx_; task < taskEnd_; task += usedCoreNum_) {
            this->SetTaskBank(task);
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);
            const uint64_t chunkStart = localChunk * bt_;
            if (chunkStart >= localT) {
                continue;
            }
            const uint64_t validRows = (chunkStart + bt_ <= localT) ? bt_ : (localT - chunkStart);
            const uint64_t tok0 = bos + chunkStart;
            const uint64_t nHvWin = (hv_ + 1ULL) / 2ULL;
            for (uint64_t w = 0; w < nHvWin; ++w) {
                const uint64_t hvBase = w * 2ULL;
                const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
                for (uint32_t iK = 0; iK < nBk; ++iK) {
                    for (uint32_t h = 0; h < headCnt; ++h) {
                        const uint64_t slot = this->SlotOf(w, h);
                        const uint64_t iHv = hvBase + h;
                        const uint64_t iH = iHv / group_;
                        GateOnlyVec(slot, bIdx, iHv, iH, tok0, localChunk, validRows, nBv, iK);
#if !USE_GATE_EARLY_SET
                        SetVGateJoined();
#endif
                    }
                    for (uint32_t h = 0; h < headCnt; ++h) {
                        const uint64_t slot = this->SlotOf(w, h);
                        const uint64_t iHv = hvBase + h;
                        const uint64_t iH = iHv / group_;
                        Catlass::Arch::CrossCoreWaitFlag(cS2_);
                        EpilogVec(slot, bIdx, iHv, iH, tok0, localChunk, validRows, nBv, iK);
                    }
                }
            }
        }
#if USE_VEC_MTE2_PP
        ReleaseVecMte2Pp();
#endif
    }

    // F6 OpC: Mask + Store.
    __aicore__ inline void ProcessStageC()
    {
        for (uint64_t task = taskBegin_ + coreIdx_; task < taskEnd_; task += usedCoreNum_) {
            this->SetTaskBank(task);
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);
            const uint64_t chunkStart = localChunk * bt_;
            if (chunkStart >= localT) {
                continue;
            }
            const uint64_t validRows = (chunkStart + bt_ <= localT) ? bt_ : (localT - chunkStart);
            const uint64_t tok0 = bos + chunkStart;
            const uint64_t nHvWin = (hv_ + 1ULL) / 2ULL;
            for (uint64_t w = 0; w < nHvWin; ++w) {
                const uint64_t hvBase = w * 2ULL;
                const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
                for (uint32_t h = 0; h < headCnt; ++h) {
                    const uint64_t slot = this->SlotOf(w, h);
                    const uint64_t iHv = hvBase + h;
                    Stage3MaskVec(slot, bIdx, iHv, validRows);
                    SetVMaskJoined();
                }
                for (uint32_t h = 0; h < headCnt; ++h) {
                    const uint64_t slot = this->SlotOf(w, h);
                    const uint64_t iHv = hvBase + h;
                    Catlass::Arch::CrossCoreWaitFlag(cS3_);
                    Stage3StoreVec(slot, bIdx, iHv, tok0, validRows);
                }
            }
        }
#if USE_VEC_MTE2_PP
        ReleaseVecMte2Pp();
#endif
    }

private:
    __aicore__ inline void RunWindowStage0VecStage(uint64_t windowIdx, uint64_t bIdx, uint64_t tok0,
                                                   uint64_t validRows, uint32_t nBv)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
#if USE_SYNC_PLAN_V1
        Catlass::Arch::CrossCoreWaitFlag(cS0_);
#endif
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = this->SlotOf(windowIdx, h);
            const uint64_t iHv = hvBase + h;
#if !USE_SYNC_PLAN_V1
            Catlass::Arch::CrossCoreWaitFlag(cS0_);
#endif
            Stage0Vec(slot, bIdx, iHv, tok0, validRows, nBv);
#if !USE_VS0_ONCE_PER_WINDOW
            SetVS0Joined();
#endif
        }
#if USE_VS0_ONCE_PER_WINDOW
        SetVS0Joined();
#endif
    }

    __aicore__ inline void RunWindowStage0Vec(uint64_t windowIdx, uint64_t bIdx, uint64_t tok0,
                                              uint64_t validRows, uint32_t nBv)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
        const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);
#if USE_SYNC_PLAN_V1
        Catlass::Arch::CrossCoreWaitFlag(cS0_);
#endif
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = winSlot ^ h;
            const uint64_t iHv = hvBase + h;
#if !USE_SYNC_PLAN_V1
            Catlass::Arch::CrossCoreWaitFlag(cS0_);
#endif
            Stage0Vec(slot, bIdx, iHv, tok0, validRows, nBv);
#if !USE_VS0_ONCE_PER_WINDOW
            SetVS0Joined();
#endif
        }
#if USE_VS0_ONCE_PER_WINDOW
        SetVS0Joined(); // one 0x2 Set per window (both AIV); Cube Wait once
#endif
    }

    __aicore__ inline void RunWindowPostBodyVec(uint64_t windowIdx, uint64_t bIdx, uint64_t tok0,
                                                uint64_t localChunk, uint64_t validRows, uint32_t nBv,
                                                uint32_t nBk)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
        const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);

        for (uint32_t iK = 0; iK < nBk; ++iK) {
#if USE_KG_GATE_INTERLEAVE
            // Per-head Kg→Gate: Set V_GATE(h0) while Cube may still be in Stage1(h1).
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
                const uint64_t iH = iHv / group_;
                KgVec(slot, bIdx, iHv, iH, tok0, validRows, iK);
                Catlass::Arch::CrossCoreWaitFlag(cS1_);
                GateOnlyVec(slot, bIdx, iHv, iH, tok0, localChunk, validRows, nBv, iK);
#if !USE_GATE_EARLY_SET
                SetVGateJoined();
#endif
            }
#else
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
                const uint64_t iH = iHv / group_;
                KgVec(slot, bIdx, iHv, iH, tok0, validRows, iK);
            }
#if USE_SYNC_PLAN_V1
            Catlass::Arch::CrossCoreWaitFlag(cS1_);
#endif
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
                const uint64_t iH = iHv / group_;
#if !USE_SYNC_PLAN_V1
                Catlass::Arch::CrossCoreWaitFlag(cS1_);
#endif
                GateOnlyVec(slot, bIdx, iHv, iH, tok0, localChunk, validRows, nBv, iK);
#if !USE_GATE_EARLY_SET
                SetVGateJoined();
#endif
            }
#endif
            for (uint32_t h = 0; h < headCnt; ++h) {
                const uint64_t slot = winSlot ^ h;
                const uint64_t iHv = hvBase + h;
                const uint64_t iH = iHv / group_;
                Catlass::Arch::CrossCoreWaitFlag(cS2_);
                EpilogVec(slot, bIdx, iHv, iH, tok0, localChunk, validRows, nBv, iK);
#if USE_EARLY_MASK_PER_HEAD && !USE_MASK_SOFT_LEAD
                if (iK + 1 == nBk) {
                    Stage3MaskVec(slot, bIdx, iHv, validRows);
                    SetVMaskJoined();
                }
#endif
            }
        }

#if !USE_EARLY_MASK_PER_HEAD && !USE_MASK_SOFT_LEAD
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = winSlot ^ h;
            const uint64_t iHv = hvBase + h;
            Stage3MaskVec(slot, bIdx, iHv, validRows);
            SetVMaskJoined();
        }
#endif
    }

    __aicore__ inline void RunWindowPostTailVec(uint64_t windowIdx, uint64_t bIdx, uint64_t tok0,
                                                uint64_t validRows)
    {
        const uint64_t hvBase = windowIdx * 2ULL;
        const uint32_t headCnt = (hvBase + 2 <= hv_) ? 2U : static_cast<uint32_t>(hv_ - hvBase);
        const uint32_t winSlot = static_cast<uint32_t>((windowIdx & 1ULL) * 2ULL);
        for (uint32_t h = 0; h < headCnt; ++h) {
            const uint64_t slot = winSlot ^ h;
            const uint64_t iHv = hvBase + h;
            Catlass::Arch::CrossCoreWaitFlag(cS3_);
            Stage3StoreVec(slot, bIdx, iHv, tok0, validRows);
            SetSlotFreeJoined(slot);
        }
    }

    __aicore__ inline void RunWindowPostVec(uint64_t windowIdx, uint64_t bIdx, uint64_t tok0,
                                            uint64_t localChunk, uint64_t validRows, uint32_t nBv, uint32_t nBk)
    {
        RunWindowPostBodyVec(windowIdx, bIdx, tok0, localChunk, validRows, nBv, nBk);
        RunWindowPostTailVec(windowIdx, bIdx, tok0, validRows);
    }

    // MIX_AIC_1_2: each AIV Sets; Cube Wait once consumes 2 credits.
    // Prefill soft-lead: Barrier before SetVS0 to avoid 0x2 skew (isub §5.1).
    __aicore__ inline void SetVGateJoined()
    {
        PipeBarrier<PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vGate_);
    }
    __aicore__ inline void SetVS0Joined()
    {
#if USE_WIN_SOFT_LEAD
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
#endif
        PipeBarrier<PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vS0_);
    }
    __aicore__ inline void SetVMaskJoined()
    {
        PipeBarrier<PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vMask_);
    }
    __aicore__ inline void SetSlotFreeJoined(uint64_t slot)
    {
        PipeBarrier<PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(slotFree_[slot]);
    }

    // DESIGN §5.1: dual-AIV contiguous half-split (PR190). db/dgk merge via
    // dbMergeWs[2]/dgkMergeWs[2]; gn stashed in dgkWs during Kg→Gate.
    static constexpr bool kSingleWriterAiv = false;

    __aicore__ inline uint32_t OwnedRowBegin() const
    {
        if (kSingleWriterAiv || subBlockNum_ <= 1) {
            return IsSub0() ? 0U : static_cast<uint32_t>(bt_);
        }
        // Invert lanes so Sub0 owns the upper half (incl. lastRow / state path).
        // Profiling: AIV1 was critical-path with state+half; Sub0 idled on merge Barrier.
        const uint32_t lane = (subBlockNum_ - 1U - subBlockIdx_);
        return lane * (static_cast<uint32_t>(bt_) / subBlockNum_);
    }
    __aicore__ inline uint32_t OwnedRowCount() const
    {
        if (kSingleWriterAiv || subBlockNum_ <= 1) {
            return IsSub0() ? static_cast<uint32_t>(bt_) : 0U;
        }
        return static_cast<uint32_t>(bt_) / subBlockNum_;
    }
    __aicore__ inline bool OwnRow(uint32_t row) const
    {
        const uint32_t begin = OwnedRowBegin();
        return row >= begin && row < begin + OwnedRowCount();
    }
    __aicore__ inline bool IsSub0() const
    {
        return subBlockIdx_ == 0 || subBlockNum_ <= 1;
    }
    __aicore__ inline bool IsComputeAiv() const
    {
        return !kSingleWriterAiv || IsSub0();
    }
    __aicore__ inline uint64_t DbMergeOff() const
    {
        const uint32_t lane = (kSingleWriterAiv || subBlockNum_ <= 1) ? 0U : subBlockIdx_;
        return SlotLayoutF32::dbMergeWs + static_cast<uint64_t>(lane) * MAX_BT;
    }
    __aicore__ inline uint64_t DgkPartialOff() const
    {
        const uint32_t lane = (kSingleWriterAiv || subBlockNum_ <= 1) ? 0U : subBlockIdx_;
        return SlotLayoutF32::dgkMergeWs + static_cast<uint64_t>(lane) * MAX_BK;
    }
    // Non-compute AIV must still join AIV↔AIV barriers inside stage funcs.
    __aicore__ inline void JoinAivBarrier()
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
    }

    // ==================== generic helpers ====================
    template <typename Elem>
    __aicore__ inline void CopyStrided(LocalTensor<Elem> &dst, GlobalTensor<Elem> &src, uint64_t base,
                                       uint64_t rowStrideElems, uint32_t rows, uint32_t cols)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        if (rowStrideElems == cols) {
            this->CopyVectorIn(dst, src, base, static_cast<uint64_t>(rows) * cols);
            return;
        }
        DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>(cols * sizeof(Elem)),
                              static_cast<uint16_t>((rowStrideElems - cols) * sizeof(Elem)), 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[base], params, padParams);
    }

    template <typename Elem>
    __aicore__ inline void CopyStridedZeroPad(LocalTensor<Elem> &dst, GlobalTensor<Elem> &src, uint64_t base,
                                              uint64_t rowStrideElems, uint32_t validRows, uint32_t rows,
                                              uint32_t cols)
    {
        if (validRows < rows) {
            Duplicate(dst, static_cast<Elem>(0), rows * cols);
            SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
        }
        if (validRows == 0) {
            return;
        }
        CopyStrided(dst, src, base, rowStrideElems, validRows, cols);
    }

    template <typename Elem>
    __aicore__ inline void CopyStridedOut(GlobalTensor<Elem> &dst, uint64_t base, uint64_t rowStrideElems,
                                          LocalTensor<Elem> &src, uint32_t rows, uint32_t cols)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        if (rowStrideElems == cols) {
            this->CopyVectorOut(dst, base, src, static_cast<uint64_t>(rows) * cols);
            return;
        }
        DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>(cols * sizeof(Elem)), 0,
                              static_cast<uint16_t>((rowStrideElems - cols) * sizeof(Elem))};
        DataCopyPad(dst[base], src, params);
    }

    // Store contiguous owned row block (PR190 half-split).
    template <typename Elem>
    // compactSrc=true: src holds owned rows at [0..nr) (Kg/Gate/Epilog BK panels under USE_OWNED_ARENA).
    // compactSrc=false: src is full [BT×cols] with owned rows at [r0..] (Stage0 partial / Mask).
    __aicore__ inline void CopyStridedOutOwned(GlobalTensor<Elem> &dst, uint64_t base, uint64_t rowStrideElems,
                                               LocalTensor<Elem> &src, uint32_t rows, uint32_t cols,
                                               bool compactSrc = false)
    {
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 >= rows || nr == 0) {
            return;
        }
        if (r0 + nr > rows) {
            nr = rows - r0;
        }
#if USE_OWNED_ARENA
        LocalTensor<Elem> srcOwned = compactSrc ? src : src[r0 * cols];
#else
        (void)compactSrc;
        LocalTensor<Elem> srcOwned = src[r0 * cols];
#endif
        CopyStridedOut(dst, base + static_cast<uint64_t>(r0) * rowStrideElems, rowStrideElems, srcOwned, nr, cols);
    }

    template <typename Elem>
    __aicore__ inline void CopyWsRowsOwned(GlobalTensor<Elem> &dst, uint64_t base, LocalTensor<Elem> &src,
                                           uint32_t rows, uint32_t cols, uint64_t rowStride)
    {
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 >= rows || nr == 0) {
            return;
        }
        if (r0 + nr > rows) {
            nr = rows - r0;
        }
#if USE_OWNED_ARENA
        // src is compact [0..nr) rows (no r0 gap).
        LocalTensor<Elem> srcOwned = src;
#else
        LocalTensor<Elem> srcOwned = src[r0 * cols];
#endif
        if (rowStride == cols) {
            DataCopy(dst[base + static_cast<uint64_t>(r0) * rowStride], srcOwned, nr * cols);
            return;
        }
        for (uint32_t i = 0; i < nr; ++i) {
            DataCopy(dst[base + static_cast<uint64_t>(r0 + i) * rowStride], srcOwned[i * cols], cols);
        }
    }

    // Cross-pipe handoffs. PipeBarrier<PIPE_V> only orders instructions *within*
    // the vector pipe — it does NOT wait for MTE2/MTE3 DMA completion, so every
    // GM/workspace DataCopy must be bracketed by one of these instead.
    __aicore__ inline void SyncMte2ToV()
    {
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
    }
    __aicore__ inline void SyncVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
    }
    // Needed before a DataCopy(GM->UB) reuses a scratch buffer that a prior V op
    // just read from (WAR hazard — MTE2 must not overwrite until V is done reading).
    __aicore__ inline void SyncVToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
    }
    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
    }

#if USE_VEC_MTE2_PP
    static constexpr uint32_t kVecMte2Pp = 2;
    __aicore__ inline void InitVecMte2Pp()
    {
        for (uint32_t i = 0; i < kVecMte2Pp; ++i) {
            mte2ToVPp_[i] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_V>());
            vToMte2Pp_[i] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE2>());
            SetFlag<HardEvent::V_MTE2>(vToMte2Pp_[i]);
        }
        ppIdx_ = 0;
        ppInited_ = true;
    }
    __aicore__ inline void ReleaseVecMte2Pp()
    {
        if (!ppInited_) {
            return;
        }
        for (uint32_t i = 0; i < kVecMte2Pp; ++i) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2Pp_[i]);
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVPp_[i]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Pp_[i]);
        }
        ppInited_ = false;
    }
    // Issue GM→UB copy on ping-pong slot; returns slot for PpAcquire.
    // GmT must be GlobalTensor / indexed GlobalTensor (not raw __gm__*).
    template <typename GmT>
    __aicore__ inline uint32_t PpCopyInF32(LocalTensor<float> dst, GmT src, uint32_t nElem)
    {
        const uint32_t idx = ppIdx_;
        WaitFlag<HardEvent::V_MTE2>(vToMte2Pp_[idx]);
        DataCopy(dst, src, nElem);
        SetFlag<HardEvent::MTE2_V>(mte2ToVPp_[idx]);
        ppIdx_ ^= 1U;
        return idx;
    }
    template <typename GmT>
    __aicore__ inline uint32_t PpCopyInT(LocalTensor<T> dst, GmT src, uint32_t nElem)
    {
        const uint32_t idx = ppIdx_;
        WaitFlag<HardEvent::V_MTE2>(vToMte2Pp_[idx]);
        DataCopy(dst, src, nElem);
        SetFlag<HardEvent::MTE2_V>(mte2ToVPp_[idx]);
        ppIdx_ ^= 1U;
        return idx;
    }
    // Begin/end around non-DataCopy loads (e.g. CopyStrided).
    __aicore__ inline uint32_t PpBeginSlot()
    {
        const uint32_t idx = ppIdx_;
        WaitFlag<HardEvent::V_MTE2>(vToMte2Pp_[idx]);
        return idx;
    }
    __aicore__ inline void PpEndSlot(uint32_t idx)
    {
        SetFlag<HardEvent::MTE2_V>(mte2ToVPp_[idx]);
        ppIdx_ ^= 1U;
    }
    __aicore__ inline void PpAcquire(uint32_t idx)
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToVPp_[idx]);
        SetFlag<HardEvent::V_MTE2>(vToMte2Pp_[idx]);
    }
#endif

    // By value: AscendC `t[offset]` yields a temporary LocalTensor handle.
    __aicore__ inline void Exp2InPlace(LocalTensor<float> t, uint32_t count)
    {
        Muls(t, t, LN2, count);
        Mins(t, t, EXP_INPUT_MAX, count);
        Maxs(t, t, EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
        Exp(t, t, count);
        PipeBarrier<PIPE_V>();
    }

    // dst[r,c] *= rowVec[r] for r in [0,rows), c in [0,cols). rows must be a
    // multiple of 8 (true for MAX_BT=64); cols tiled in 8-wide chunks.
    // brcbScratch must hold >= rows*8 floats (Brcb expands each row's scalar into
    // its own 8-float block; the Mul below advances src1 by one block per row).
    __aicore__ inline void RowBroadcastMulInPlace(LocalTensor<float> &dst, LocalTensor<float> &rowVec,
                                                  LocalTensor<float> &brcbScratch, uint32_t rows, uint32_t cols)
    {
        const uint8_t brcbRepeat = static_cast<uint8_t>((rows + 7) / 8);
        Brcb(brcbScratch, rowVec, brcbRepeat, {1, 8});
        PipeBarrier<PIPE_V>();
        const uint8_t rowBlk = static_cast<uint8_t>((cols * sizeof(float)) / 32);
        for (uint32_t col = 0; col < cols; col += 8) {
            Mul(dst[col], dst[col], brcbScratch, static_cast<uint64_t>(8), static_cast<uint8_t>(rows),
                {1, 1, 0, rowBlk, rowBlk, 1});
        }
        PipeBarrier<PIPE_V>();
    }

    // dst[r,c] *= colVec[c] for r in [0,rows), c in [0,cols<=64) (colVec is the
    // shared "row" replicated for every output row via repStride=0).
    __aicore__ inline void ColBroadcastMulInPlace(LocalTensor<float> &dst, LocalTensor<float> &colVec, uint32_t rows,
                                                  uint32_t cols)
    {
        const uint8_t rowBlk = static_cast<uint8_t>((cols * sizeof(float)) / 32);
        Mul(dst, dst, colVec, static_cast<uint64_t>(cols), static_cast<uint8_t>(rows), {1, 1, 1, rowBlk, rowBlk, 0});
        PipeBarrier<PIPE_V>();
    }

    // Reduce mat[rows,width] along the last axis; add per-row sums into acc[rows].
    // Prefer WholeReduceSum (PR190) when width is a multiple of 64; else vector-fold
    // to 8 then one WholeReduceSum per row (no GetValue/SetValue scalar tail).
    __aicore__ inline void RowFoldSumAddInto(LocalTensor<float> &acc, LocalTensor<float> &mat, uint32_t rows,
                                             uint32_t width)
    {
        if (rows == 0 || width == 0) {
            return;
        }
        LocalTensor<float> reduceScratch = brcbBuf_.Get<float>(); // >= rows*8
        if ((width % 64U) == 0U) {
            const uint32_t n64 = width / 64U;
            for (uint32_t r = 0; r < rows; ++r) {
                WholeReduceSum(reduceScratch[r * 8], mat[r * width], 64, static_cast<uint8_t>(n64), 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
            // Compact per-row sums into reduceScratch[0..rows).
            // In-place safe: write dst[r] while reading src[r*8 ..] (r < r*8 for r>0).
            WholeReduceSum(reduceScratch, reduceScratch, static_cast<uint8_t>(n64), static_cast<uint8_t>(rows), 1, 1,
                           1);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, reduceScratch, rows);
            PipeBarrier<PIPE_V>();
            return;
        }
        // Generic: in-place fold width → 8, then WholeReduceSum(8) → 1 per row.
        uint32_t remain = width;
        const uint8_t rowBlk = static_cast<uint8_t>((width * sizeof(float) + 31) / 32);
        while (remain > 8) {
            const uint32_t calcCnt = remain / 2;
            const uint32_t newRemain = remain - calcCnt;
            Add(mat, mat, mat[newRemain], calcCnt, static_cast<uint8_t>(rows), {1, 1, 1, rowBlk, rowBlk, rowBlk});
            PipeBarrier<PIPE_V>();
            remain = newRemain;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            WholeReduceSum(reduceScratch[r * 8], mat[r * width], static_cast<uint8_t>(remain), 1, 1, 1, 1);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduceScratch, reduceScratch, 1, static_cast<uint8_t>(rows), 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, reduceScratch, rows);
        PipeBarrier<PIPE_V>();
    }

    // Reduce mat[rows,width] along the leading axis into acc[width].
    // Pairwise row tree (~log rows vector Adds) instead of rows serial Adds.
    __aicore__ inline void ColSumAddInto(LocalTensor<float> &acc, LocalTensor<float> &mat, uint32_t rows,
                                         uint32_t width)
    {
        if (rows == 0 || width == 0) {
            return;
        }
        uint32_t n = rows;
        while (n > 1) {
            const uint32_t half = n >> 1;
            Add(mat, mat, mat[half * width], half * width);
            PipeBarrier<PIPE_V>();
            if ((n & 1U) != 0U) {
                Add(mat, mat, mat[(n - 1) * width], width);
                PipeBarrier<PIPE_V>();
            }
            n = half;
        }
        Add(acc, acc, mat, width);
#if !USE_FOLD_BAR_SLIM
        PipeBarrier<PIPE_V>();
#endif
    }

    // Column-reduce only rows owned by this AIV (contiguous half).
    // mat is compact owned rows at [0..nr) when USE_OWNED_ARENA (Gate dkGateExp); else full with r0 gap.
    __aicore__ inline void ColSumAddIntoOwned(LocalTensor<float> &acc, LocalTensor<float> &mat, uint32_t rows,
                                              uint32_t width)
    {
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 >= rows || nr == 0) {
            return;
        }
        if (r0 + nr > rows) {
            nr = rows - r0;
        }
#if USE_OWNED_ARENA
        LocalTensor<float> matOwned = mat;
#else
        LocalTensor<float> matOwned = mat[r0 * width];
#endif
        ColSumAddInto(acc, matOwned, nr, width);
    }

    // Full-panel mat with r0 gap (Stage0 partial). Compact callers use RowFoldSumAddInto directly.
    __aicore__ inline void RowFoldSumAddIntoOwned(LocalTensor<float> &acc, LocalTensor<float> &mat, uint32_t rows,
                                                  uint32_t width)
    {
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 >= rows || nr == 0) {
            return;
        }
        if (r0 + nr > rows) {
            nr = rows - r0;
        }
        Duplicate(acc, 0.0f, static_cast<uint32_t>(bt_));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> matOwned = mat[r0 * width];
        LocalTensor<float> accOwned = acc[r0];
        RowFoldSumAddInto(accOwned, matOwned, nr, width);
    }

    __aicore__ inline void MergeDbAcc(uint64_t slot)
    {
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        if (kSingleWriterAiv || subBlockNum_ <= 1) {
            if (IsSub0()) {
                DataCopy(dbAcc, wsF32_[f32Base + SlotLayoutF32::dbMergeWs], static_cast<uint32_t>(bt_));
                SyncMte2ToV();
            }
            return;
        }
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
        if (!IsSub0()) {
            return;
        }
        LocalTensor<float> tmp = arenaF32_.Get<float>();
        Duplicate(dbAcc, 0.0f, static_cast<uint32_t>(bt_));
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < subBlockNum_ && s < 2; ++s) {
            DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dbMergeWs + static_cast<uint64_t>(s) * MAX_BT],
                     static_cast<uint32_t>(bt_));
            SyncMte2ToV();
            Add(dbAcc, dbAcc, tmp, static_cast<uint32_t>(bt_));
            PipeBarrier<PIPE_V>();
            if (s + 1 < subBlockNum_) {
                SyncVToMte2();
            }
        }
    }

    __aicore__ inline void MergeDgkAcc(uint64_t slot, LocalTensor<float> &dgkAcc, uint32_t bk)
    {
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        if (kSingleWriterAiv || subBlockNum_ <= 1) {
            if (IsComputeAiv()) {
                SyncVToMte3();
                DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkWs], dgkAcc, bk);
                // Epilog also reads dgkMergeWs[1] in legacy single-writer layout.
                DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], dgkAcc, bk);
                SyncMte3ToMte2();
            }
            return;
        }
        // Partials in dgkMergeWs[lane]; merged result → dgkWs (gn was in dgkWs, already consumed).
        SyncVToMte3();
        DataCopy(wsF32_[f32Base + DgkPartialOff()], dgkAcc, bk);
        SyncMte3ToMte2();
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
        if (!IsSub0()) {
            return;
        }
        LocalTensor<float> tmp = brcbBuf_.Get<float>();
        Duplicate(dgkAcc, 0.0f, bk);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < subBlockNum_ && s < 2; ++s) {
            DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + static_cast<uint64_t>(s) * MAX_BK], bk);
            SyncMte2ToV();
            Add(dgkAcc, dgkAcc, tmp, bk);
            PipeBarrier<PIPE_V>();
            if (s + 1 < subBlockNum_) {
                SyncVToMte2();
            }
        }
        SyncVToMte3();
        DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkWs], dgkAcc, bk);
        DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], dgkAcc, bk);
        SyncMte3ToMte2();
    }

    // Partial-chunk A@dv in UB (Cube Stage0 skips A@dv when validRows < BT).
    // Vector path: Brcb(A[r,k]) × dv[k,:] + Acc — no GetValue/SetValue scalar sync.
    __aicore__ inline void ComputeDvbPartial(LocalTensor<float> &dvb, uint64_t bIdx, uint64_t iHv, uint64_t tok0,
                                             uint64_t validRows, uint64_t vOff, uint32_t bv)
    {
        const uint32_t vr = static_cast<uint32_t>(validRows);
        const uint32_t aElems = vr * vr;
        const uint32_t dvElems = vr * bv;
        LocalTensor<T> tIn = arenaT_.Get<T>();
        LocalTensor<T> aIn = tIn;
        LocalTensor<T> dvInT = tIn[aElems];
        CopyStrided(aIn, a_, this->AOff(bIdx, iHv, tok0, 0), bt_, vr, vr);
        CopyStrided(dvInT, dvIn_, this->HvVOff(bIdx, iHv, tok0, vOff), vDim_, vr, bv);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        LocalTensor<float> aFp = arenaF32_.Get<float>()[static_cast<uint32_t>(bt_) * bv];
        LocalTensor<float> dvFp = aFp[aElems];
        Cast(aFp, aIn, RoundMode::CAST_NONE, aElems);
        Cast(dvFp, dvInT, RoundMode::CAST_NONE, dvElems);
        PipeBarrier<PIPE_V>();
        Duplicate(dvb, 0.0f, static_cast<uint32_t>(bt_) * bv);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> rowTmp = dvFp[dvElems];
        LocalTensor<float> brcbScratch = brcbBuf_.Get<float>();
        // dvb[r,:] += A[r,k] * dv[k,:]
        for (uint32_t kk = 0; kk < vr; ++kk) {
            LocalTensor<float> dvRow = dvFp[kk * bv];
            for (uint32_t r = 0; r < vr; ++r) {
                Brcb(brcbScratch, aFp[r * vr + kk], 1, {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t c = 0; c < bv; c += 8) {
                    const uint32_t n = (c + 8 <= bv) ? 8U : (bv - c);
                    if (n == 8) {
                        Mul(rowTmp[c], dvRow[c], brcbScratch, 8);
                    } else {
                        Mul(rowTmp[c], dvRow[c], brcbScratch, n);
                    }
                }
                PipeBarrier<PIPE_V>();
                Add(dvb[r * bv], dvb[r * bv], rowTmp, bv);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // ==================== Stage0: dv2/db + dAWs init ====================
    __aicore__ inline void Stage0Vec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t validRows,
                                     uint32_t nBv)
    {
        if (!IsComputeAiv()) {
            JoinAivBarrier(); // mirrors mid-Stage0 barrier after dAWs
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const bool partialBt = (validRows < bt_);

        LocalTensor<float> beta = betaBuf_.Get<float>();
        LocalTensor<T> betaIn = arenaT_.Get<T>();
        CopyStridedZeroPad(betaIn, beta_, this->BetaOff(bIdx, iHv, tok0), 1, static_cast<uint32_t>(validRows),
                          static_cast<uint32_t>(bt_), 1);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(beta, betaIn, RoundMode::CAST_NONE, static_cast<uint32_t>(bt_));
        PipeBarrier<PIPE_V>();
        // Park beta in per-slot ws — window Stage0 for head1 must not clobber head0.
        SyncVToMte3();
        DataCopy(wsF32_[f32Base + SlotLayoutF32::betaWs], beta, static_cast<uint32_t>(bt_));
        SyncMte3ToMte2();

        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
        Duplicate(dbAcc, 0.0f, static_cast<uint32_t>(bt_));
        PipeBarrier<PIPE_V>();

        if (IsSub0()) {
#if USE_STAGE0_DA_L0C_ACCUM
            // Cube already wrote reduced dA into dAWs — nothing to Σ.
            (void)nBv;
#else
            LocalTensor<float> dAWs = arenaF32_.Get<float>();
            const uint32_t btbt = static_cast<uint32_t>(bt_ * bt_);
            DataCopy(dAWs, wsF32_[f32Base + SlotLayoutF32::dASlot], btbt);
            SyncMte2ToV();
            for (uint32_t iv = 1; iv < nBv; ++iv) {
                LocalTensor<float> tmp = arenaF32_.Get<float>()[btbt];
                DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dASlot + static_cast<uint64_t>(iv) * MAX_BT * MAX_BT],
                         btbt);
                SyncMte2ToV();
                Add(dAWs, dAWs, tmp, btbt);
                PipeBarrier<PIPE_V>();
            }
            SyncVToMte3();
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dAWs], dAWs, btbt);
            SyncMte3ToMte2();
#endif
        }
#if !(USE_MERGE_BARRIER_ONLY && USE_STAGE0_DA_L0C_ACCUM)
        // Without L0C accum: Sub0 Σ dASlot→dAWs; both AIVs must wait before owned dv2.
        // With L0C accum + merge-only: no shared write here — skip Join.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
#endif

        for (uint32_t iv = 0; iv < nBv; ++iv) {
            const uint32_t bv = this->BvSize(iv);
            const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
            const uint32_t elems = static_cast<uint32_t>(bt_) * bv;
            const uint32_t r0 = OwnedRowBegin();
            uint32_t nr = OwnedRowCount();
            if (r0 >= static_cast<uint32_t>(bt_) || nr == 0) {
                continue;
            }
            if (r0 + nr > static_cast<uint32_t>(bt_)) {
                nr = static_cast<uint32_t>(bt_) - r0;
            }
            // Partial chunks keep full-panel path (validRows may cut mid-half).
            if (partialBt) {
                LocalTensor<float> dvb = arenaF32_.Get<float>();
                ComputeDvbPartial(dvb, bIdx, iHv, tok0, validRows, vOff, bv);
                LocalTensor<T> vIn = arenaT_.Get<T>();
                CopyStridedZeroPad(vIn, v_, this->HvVOff(bIdx, iHv, tok0, vOff), vDim_,
                                   static_cast<uint32_t>(validRows), static_cast<uint32_t>(bt_), bv);
                SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
                WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
                LocalTensor<float> vFp = arenaF32_.Get<float>()[elems];
                Cast(vFp, vIn, RoundMode::CAST_NONE, elems);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> dv2Fp = arenaF32_.Get<float>()[2 * elems];
                DataCopy(dv2Fp, dvb, elems);
                SyncMte2ToV();
                LocalTensor<float> brcbScratch = brcbBuf_.Get<float>();
                RowBroadcastMulInPlace(dv2Fp, beta, brcbScratch, static_cast<uint32_t>(bt_), bv);
                LocalTensor<float> prod = arenaF32_.Get<float>()[3 * elems];
                Mul(prod, dvb, vFp, elems);
                PipeBarrier<PIPE_V>();
                RowFoldSumAddIntoOwned(dbAcc, prod, static_cast<uint32_t>(validRows), bv);
                LocalTensor<T> dv2T = arenaT_.Get<T>();
                Cast(dv2T, dv2Fp, RoundMode::CAST_RINT, elems);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                CopyStridedOutOwned(dv2_, this->HvVOff(bIdx, iHv, tok0, vOff), vDim_, dv2T,
                                    static_cast<uint32_t>(validRows), bv);
                SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                continue;
            }
            const uint32_t ownedElems = nr * bv;
            LocalTensor<float> dvb = arenaF32_.Get<float>();
            const uint64_t dvbPanel =
                f32Base + SlotLayoutF32::dvbWs + static_cast<uint64_t>(iv) * MAX_BT * MAX_BV;
            CopyStrided(dvb, wsF32_, dvbPanel + static_cast<uint64_t>(r0) * bv, bv, nr, bv);
            SyncMte2ToV();

            LocalTensor<T> vIn = arenaT_.Get<T>();
            CopyStrided(vIn, v_, this->HvVOff(bIdx, iHv, tok0 + r0, vOff), vDim_, nr, bv);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            LocalTensor<float> vFp = arenaF32_.Get<float>()[ownedElems];
            Cast(vFp, vIn, RoundMode::CAST_NONE, ownedElems);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> dv2Fp = arenaF32_.Get<float>()[2 * ownedElems];
            DataCopy(dv2Fp, dvb, ownedElems);
            SyncMte2ToV();
            LocalTensor<float> betaOwned = beta[r0];
            LocalTensor<float> brcbScratch = brcbBuf_.Get<float>();
            RowBroadcastMulInPlace(dv2Fp, betaOwned, brcbScratch, nr, bv);

            LocalTensor<float> prod = arenaF32_.Get<float>()[3 * ownedElems];
            Mul(prod, dvb, vFp, ownedElems);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> dbOwned = dbAcc[r0];
            RowFoldSumAddInto(dbOwned, prod, nr, bv);

            LocalTensor<T> dv2T = arenaT_.Get<T>();
            Cast(dv2T, dv2Fp, RoundMode::CAST_RINT, ownedElems);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyStridedOut(dv2_, this->HvVOff(bIdx, iHv, tok0 + r0, vOff), vDim_, dv2T, nr, bv);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        }
        // Park per-AIV db lane for Epilog / Stage3 merge (no extra barrier — AIV1
        // early-out in single-writer must not leave a dangling Wait).
        SyncVToMte3();
        DataCopy(wsF32_[f32Base + DbMergeOff()], dbAcc, static_cast<uint32_t>(bt_));
        SyncMte3ToMte2();
    }

    // ==================== Stage1 Kg (∥ Cube KvAcc, no CrossCore wait) ====================
    __aicore__ inline void KgVec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t iH, uint64_t tok0,
                                 uint64_t validRows, uint32_t iK)
    {
        if (!IsComputeAiv()) {
            JoinAivBarrier();
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const uint64_t tBase = slot * SlotLayoutT::TOTAL;
        const uint32_t bk = this->BkSize(iK);
        const uint64_t kOff = static_cast<uint64_t>(iK) * MAX_BK;
#if USE_OWNED_ARENA
        const uint32_t btbk = ARENA_BT_ROWS * bk;
#else
        const uint32_t btbk = static_cast<uint32_t>(bt_) * bk;
#endif

        LocalTensor<float> arena = arenaF32_.Get<float>();
        LocalTensor<float> kFp = arena;
        LocalTensor<float> gFp = arena[btbk];
        LocalTensor<float> gkExp = arena[2 * btbk];
        LocalTensor<float> kgFp = arena[3 * btbk];

        LocalTensor<T> tIn = arenaT_.Get<T>();
        LocalTensor<T> kIn = tIn;
        LocalTensor<T> gIn = tIn[btbk];
        LocalTensor<T> gnIn = tIn[2 * btbk];

        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 >= static_cast<uint32_t>(bt_) || nr == 0) {
            if (IsSub0()) {
                // Park raw gn → dgkWs; optional exp(gn) → dgkMergeWs for Epilog.
                this->CopyVectorIn(gnIn, g_, this->HvKOff(bIdx, iHv, tok0 + validRows - 1, kOff), bk);
                SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
                WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
                LocalTensor<float> gnFp = smallBuf_.Get<float>();
                Cast(gnFp, gnIn, RoundMode::CAST_NONE, bk);
                PipeBarrier<PIPE_V>();
                SyncVToMte3();
                DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkWs], gnFp, bk);
#if USE_EXP_GN_PARK
                SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
                WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
                Exp2InPlace(gnFp, bk);
                SyncVToMte3();
                DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkMergeWs], gnFp, bk);
#endif
                SyncMte3ToMte2();
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            PipeBarrier<PIPE_MTE3>();
            return;
        }
        if (r0 + nr > static_cast<uint32_t>(bt_)) {
            nr = static_cast<uint32_t>(bt_) - r0;
        }
        const uint32_t nElem = nr * bk;
#if USE_OWNED_ARENA
        const uint32_t rowBase = 0;
#else
        const uint32_t rowBase = r0 * bk;
#endif
        const uint32_t validOwned = (r0 >= validRows) ? 0U :
            ((r0 + nr > validRows) ? static_cast<uint32_t>(validRows - r0) : nr);

        if (validOwned > 0) {
            CopyStrided(kIn, k_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
            CopyStrided(gIn, g_, this->HvKOff(bIdx, iHv, tok0 + r0, kOff), kDim_, validOwned, bk);
        }
        if (validOwned < nr) {
            Duplicate(kIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
            Duplicate(gIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
            SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
        }
        const uint64_t gnTok = tok0 + validRows - 1;
        this->CopyVectorIn(gnIn, g_, this->HvKOff(bIdx, iHv, gnTok, kOff), bk);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        LocalTensor<float> kOwned = kFp[rowBase];
        LocalTensor<float> gOwned = gFp[rowBase];
        Cast(kOwned, kIn, RoundMode::CAST_NONE, nElem);
        Cast(gOwned, gIn, RoundMode::CAST_NONE, nElem);
        LocalTensor<float> gnFp = smallBuf_.Get<float>();
        Cast(gnFp, gnIn, RoundMode::CAST_NONE, bk);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> gkOwned = gkExp[rowBase];
        LocalTensor<float> kgOwned = kgFp[rowBase];
        DataCopy(gkOwned, gOwned, nElem);
        SyncMte2ToV();
        Exp2InPlace(gkOwned, nElem);
        Mul(kgOwned, kOwned, gkOwned, nElem);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> kgT = arenaT_.Get<T>()[3 * btbk];
        LocalTensor<T> kgTOwned = kgT[rowBase];
        Cast(kgTOwned, kgOwned, RoundMode::CAST_RINT, nElem);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        if (IsSub0()) {
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkWs], gnFp, bk);
#if USE_EXP_GN_PARK
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            Exp2InPlace(gnFp, bk);
            SyncVToMte3();
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkMergeWs], gnFp, bk);
#endif
        }
#if USE_GATE_REUSE_KG_WS
        // Park owned k/g fp32 so Gate skips GM reload + Cast.
        CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::kParkWs, kFp, static_cast<uint32_t>(bt_), bk, bk);
        CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::gParkWs, gFp, static_cast<uint32_t>(bt_), bk, bk);
#endif
        CopyWsRowsOwned(wsT_, tBase + SlotLayoutT::kgWs, kgT, static_cast<uint32_t>(bt_), bk, bk);
        CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::gkWs, gkExp, static_cast<uint32_t>(bt_), bk, bk);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#if !USE_MERGE_BARRIER_ONLY
        // Owned-row parks are lane-private; Gate reads own rows / Sub0 gn locally.
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
#endif
    }

    // ==================== Stage2 gate (after Wait C_S1) ====================
    __aicore__ inline void GateOnlyVec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t iH, uint64_t tok0,
                                       uint64_t localChunk, uint64_t validRows, uint32_t nBv, uint32_t iK)
    {
        if (!IsComputeAiv()) {
            JoinAivBarrier();
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const uint64_t tBase = slot * SlotLayoutT::TOTAL;
        const uint32_t bk = this->BkSize(iK);
        const uint64_t kOff = static_cast<uint64_t>(iK) * MAX_BK;
#if USE_OWNED_ARENA
        const uint32_t btbk = ARENA_BT_ROWS * bk;
#else
        const uint32_t btbk = static_cast<uint32_t>(bt_) * bk;
#endif

        LocalTensor<float> arena = arenaF32_.Get<float>();
        LocalTensor<float> kFp = arena;
        LocalTensor<float> gFp = arena[btbk];
        LocalTensor<float> gkExp = arena[2 * btbk];
        LocalTensor<float> dqAcc = arena[3 * btbk];
        LocalTensor<float> dkAcc = arena[4 * btbk];
        LocalTensor<float> dwAcc = arena[5 * btbk];
        LocalTensor<float> scratch = arena[6 * btbk];

#if !USE_GATE_REUSE_KG_WS
        LocalTensor<T> tIn = arenaT_.Get<T>();
        LocalTensor<T> kIn = tIn;
        LocalTensor<T> gIn = tIn[btbk];
#endif

        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 + nr > static_cast<uint32_t>(bt_)) {
            nr = static_cast<uint32_t>(bt_) - r0;
        }
        const uint32_t nElem = nr * bk;
#if USE_OWNED_ARENA
        const uint32_t rowBase = 0;
#else
        const uint32_t rowBase = r0 * bk;
#endif
        const uint32_t validOwned = (r0 >= validRows) ? 0U :
            ((r0 + nr > validRows) ? static_cast<uint32_t>(validRows - r0) : nr);

#if USE_GATE_REUSE_KG_WS
        // Load parked k/g + gkWs owned halves (no GM k_/g_ reload).
        if (nr > 0) {
#if USE_VEC_MTE2_PP
            const uint32_t sK = PpCopyInF32(kFp[rowBase],
                                            wsF32_[f32Base + SlotLayoutF32::kParkWs + static_cast<uint64_t>(r0) * bk],
                                            nElem);
            const uint32_t sG = PpCopyInF32(gFp[rowBase],
                                            wsF32_[f32Base + SlotLayoutF32::gParkWs + static_cast<uint64_t>(r0) * bk],
                                            nElem);
            PpAcquire(sK);
            const uint32_t sGk = PpCopyInF32(
                gkExp[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk], nElem);
            PpAcquire(sG);
            PpAcquire(sGk);
#else
            DataCopy(kFp[rowBase], wsF32_[f32Base + SlotLayoutF32::kParkWs + static_cast<uint64_t>(r0) * bk],
                     nElem);
            DataCopy(gFp[rowBase], wsF32_[f32Base + SlotLayoutF32::gParkWs + static_cast<uint64_t>(r0) * bk],
                     nElem);
            DataCopy(gkExp[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk],
                     nElem);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
#endif
        }
        LocalTensor<float> kOwned = (nr > 0) ? kFp[rowBase] : kFp;
        LocalTensor<float> gOwned = (nr > 0) ? gFp[rowBase] : gFp;
        LocalTensor<float> gkOwned = (nr > 0) ? gkExp[rowBase] : gkExp;
        (void)validOwned;
#else
        if (validOwned > 0) {
            CopyStrided(kIn, k_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
            CopyStrided(gIn, g_, this->HvKOff(bIdx, iHv, tok0 + r0, kOff), kDim_, validOwned, bk);
        }
        if (nr > 0 && validOwned < nr) {
            Duplicate(kIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
            Duplicate(gIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
            SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
        }
        // gkWs row-split complete after Kg barrier — load owned half only.
        if (nr > 0) {
            DataCopy(gkExp[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk], nElem);
        }
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        LocalTensor<float> kOwned = (nr > 0) ? kFp[rowBase] : kFp;
        LocalTensor<float> gOwned = (nr > 0) ? gFp[rowBase] : gFp;
        LocalTensor<float> gkOwned = (nr > 0) ? gkExp[rowBase] : gkExp;
        if (nr > 0) {
            Cast(kOwned, kIn, RoundMode::CAST_NONE, nElem);
            Cast(gOwned, gIn, RoundMode::CAST_NONE, nElem);
        }
        PipeBarrier<PIPE_V>();
#endif

        // gn parked by Kg into dgkWs (pre-exp).
        LocalTensor<float> gnFp = smallBuf_.Get<float>()[MAX_BK];
#if USE_VEC_MTE2_PP
        {
            const uint32_t sGn = PpCopyInF32(gnFp, wsF32_[f32Base + SlotLayoutF32::dgkWs], bk);
            PpAcquire(sGn);
        }
#else
        DataCopy(gnFp, wsF32_[f32Base + SlotLayoutF32::dgkWs], bk);
        SyncMte2ToV();
#endif

        LocalTensor<float> dgkAcc = smallBuf_.Get<float>();
        Duplicate(dgkAcc, 0.0f, bk);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dqOwned = (nr > 0) ? dqAcc[rowBase] : dqAcc;
        LocalTensor<float> dkOwned = (nr > 0) ? dkAcc[rowBase] : dkAcc;
        LocalTensor<float> dwOwned = (nr > 0) ? dwAcc[rowBase] : dwAcc;
        if (nr > 0) {
#if USE_STAGE1_L0C_ACCUM
            // Cube already reduced across V-tiles into plane 0.
            const uint64_t off0 = static_cast<uint64_t>(r0) * bk;
#if USE_VEC_MTE2_PP
            // Soft-lead: Copy(dk)∥… while dq in flight; Mul(dq)∥Copy(dw).
            const uint32_t sDq = PpCopyInF32(dqOwned, wsF32_[f32Base + SlotLayoutF32::dqSlot + off0], nElem);
            const uint32_t sDk = PpCopyInF32(dkOwned, wsF32_[f32Base + SlotLayoutF32::dkSlot + off0], nElem);
            PpAcquire(sDq);
            const uint32_t sDw = PpCopyInF32(dwOwned, wsF32_[f32Base + SlotLayoutF32::dwSlot + off0], nElem);
            Mul(dqOwned, dqOwned, gkOwned, nElem);
            Muls(dqOwned, dqOwned, scale_, nElem);
            PpAcquire(sDk);
            PpAcquire(sDw);
#else
            // Keep per-copy MTE2↔V sync — fused triple-copy hung model msprof.
            DataCopy(dqOwned, wsF32_[f32Base + SlotLayoutF32::dqSlot + off0], nElem);
            SyncMte2ToV();
            SyncVToMte2();
            DataCopy(dkOwned, wsF32_[f32Base + SlotLayoutF32::dkSlot + off0], nElem);
            SyncMte2ToV();
            SyncVToMte2();
            DataCopy(dwOwned, wsF32_[f32Base + SlotLayoutF32::dwSlot + off0], nElem);
            SyncMte2ToV();
            Mul(dqOwned, dqOwned, gkOwned, nElem);
            Muls(dqOwned, dqOwned, scale_, nElem);
#endif
#else
            Duplicate(dqOwned, 0.0f, nElem);
            Duplicate(dkOwned, 0.0f, nElem);
            Duplicate(dwOwned, 0.0f, nElem);
            PipeBarrier<PIPE_V>();
            for (uint32_t iv = 0; iv < nBv; ++iv) {
                const uint64_t off = static_cast<uint64_t>(iv) * MAX_BT * MAX_BK +
                                    static_cast<uint64_t>(r0) * bk;
                LocalTensor<float> tmp = scratch;
                DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dqSlot + off], nElem);
                SyncMte2ToV();
                Add(dqOwned, dqOwned, tmp, nElem);
                PipeBarrier<PIPE_V>();
                SyncVToMte2();
                DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dkSlot + off], nElem);
                SyncMte2ToV();
                Add(dkOwned, dkOwned, tmp, nElem);
                PipeBarrier<PIPE_V>();
                SyncVToMte2();
                DataCopy(tmp, wsF32_[f32Base + SlotLayoutF32::dwSlot + off], nElem);
                SyncMte2ToV();
                Add(dwOwned, dwOwned, tmp, nElem);
                PipeBarrier<PIPE_V>();
                if (iv + 1 < nBv) {
                    SyncVToMte2();
                }
            }
            Mul(dqOwned, dqOwned, gkOwned, nElem);
            Muls(dqOwned, dqOwned, scale_, nElem);
#endif
        }
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
        CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::dqGatedWs, dqAcc, static_cast<uint32_t>(bt_), bk, bk);
        CopyStridedOutOwned(dq_, this->HvKOff(bIdx, iHv, tok0, kOff), kDim_, dqAcc,
                            static_cast<uint32_t>(validRows), bk, /*compactSrc=*/true);
        SyncMte3ToMte2();

        LocalTensor<float> dkGateExp = scratch;
        if (nr > 0) {
            LocalTensor<float> expOwned = dkGateExp[rowBase];
            const uint8_t rowBlk = static_cast<uint8_t>((bk * sizeof(float)) / 32);
            Sub(expOwned, gnFp, gOwned, static_cast<uint64_t>(bk), static_cast<uint8_t>(nr),
                {1, 1, 1, rowBlk, 0, rowBlk});
            PipeBarrier<PIPE_V>();
            Exp2InPlace(expOwned, nElem);
            Mul(dkOwned, dkOwned, expOwned, nElem);
            PipeBarrier<PIPE_V>();
        }
        SyncVToMte3();
        CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::dkPartialWs, dkAcc, static_cast<uint32_t>(bt_), bk, bk);
        SyncMte3ToMte2();

        if (nr > 0) {
            LocalTensor<float> kdkOwned = dkGateExp[rowBase];
            Mul(kdkOwned, kOwned, dkOwned, nElem);
            PipeBarrier<PIPE_V>();
        }
        ColSumAddIntoOwned(dgkAcc, dkGateExp, static_cast<uint32_t>(validRows), bk);
#if USE_FOLD_BAR_SLIM
        PipeBarrier<PIPE_V>();
#endif

        // dwNeg first — Cube Stage2 only needs dwNeg/kg. Do not wait on dgk merge.
        if (nr > 0) {
            Muls(dwOwned, dwOwned, -1.0f, nElem);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> dwNegT = arenaT_.Get<T>()[4 * btbk];
            LocalTensor<T> dwNegOwned = dwNegT[rowBase];
            Cast(dwNegOwned, dwOwned, RoundMode::CAST_RINT, nElem);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyWsRowsOwned(wsT_, tBase + SlotLayoutT::dwNegWs, dwNegT, static_cast<uint32_t>(bt_), bk, bk);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#if USE_GATE_EARLY_SET
        // Wake Cube Stage2 before AIV↔AIV dgk merge (overlaps merge ∥ Stage2).
        SetVGateJoined();
#endif
        MergeDgkAcc(slot, dgkAcc, bk);
#if !USE_EXP_GN_PARK
        if (IsSub0()) {
            SyncVToMte3();
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dgkMergeWs], gnFp, bk);
            SyncMte3ToMte2();
        }
#endif
#if !USE_GATE_EARLY_SET
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
#endif
    }

    __aicore__ inline void EpilogVec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t iH, uint64_t tok0,
                                     uint64_t localChunk, uint64_t validRows, uint32_t nBv, uint32_t iK)
    {
        if (!IsComputeAiv()) {
            JoinAivBarrier();
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const uint64_t tBase = slot * SlotLayoutT::TOTAL;
        const uint32_t bk = this->BkSize(iK);
        const uint64_t kOff = static_cast<uint64_t>(iK) * MAX_BK;
#if USE_OWNED_ARENA
        const uint32_t btbk = ARENA_BT_ROWS * bk;
#else
        const uint32_t btbk = static_cast<uint32_t>(bt_) * bk;
#endif

        LocalTensor<float> arena = arenaF32_.Get<float>();
        LocalTensor<float> dkgb = arena;
        LocalTensor<float> kgFp = arena[btbk];
        LocalTensor<float> qFp = arena[2 * btbk];
        LocalTensor<float> kFp = arena[3 * btbk];
        LocalTensor<float> dqWs = arena[4 * btbk];
        LocalTensor<float> dkWs = arena[5 * btbk];
        LocalTensor<float> gkWs = arena[6 * btbk];
        LocalTensor<float> scratch = arena[7 * btbk];

        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 + nr > static_cast<uint32_t>(bt_)) {
            nr = static_cast<uint32_t>(bt_) - r0;
        }
        const uint32_t nElem = (nr > 0) ? nr * bk : 0U;
        const uint32_t validOwned = (nr == 0 || r0 >= validRows) ? 0U :
            ((r0 + nr > validRows) ? static_cast<uint32_t>(validRows - r0) : nr);
#if USE_OWNED_ARENA
        const uint32_t rowBase = 0;
#else
        const uint32_t rowBase = r0 * bk;
#endif

        // Load owned halves only (dq already stored by Gate — skip re-store).
        if (nr > 0) {
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            LocalTensor<T> kgIn = arenaT_.Get<T>();
            const uint32_t sDkgb = PpCopyInF32(
                dkgb[rowBase], wsF32_[f32Base + SlotLayoutF32::dkgbWs + static_cast<uint64_t>(r0) * bk], nElem);
            const uint32_t sKg = PpCopyInT(
                kgIn, wsT_[tBase + SlotLayoutT::kgWs + static_cast<uint64_t>(r0) * bk], nElem);
            PpAcquire(sDkgb);
            PpAcquire(sKg);
            Cast(kgFp[rowBase], kgIn, RoundMode::CAST_NONE, nElem);
            PipeBarrier<PIPE_V>();

            LocalTensor<T> qIn = arenaT_.Get<T>()[btbk];
            LocalTensor<T> kIn = arenaT_.Get<T>()[2 * btbk];
            uint32_t sQ = 0;
            uint32_t sKstr = 0;
            if (validOwned > 0) {
                sQ = PpBeginSlot();
                CopyStrided(qIn, q_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
                PpEndSlot(sQ);
                sKstr = PpBeginSlot();
                CopyStrided(kIn, k_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
                PpEndSlot(sKstr);
            }
            if (validOwned < nr) {
                if (validOwned > 0) {
                    PpAcquire(sQ);
                    PpAcquire(sKstr);
                }
                Duplicate(qIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
                Duplicate(kIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
                SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
                WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
            } else if (validOwned > 0) {
                // Soft-lead: start dk while q/k still in flight.
                const uint32_t sDk = PpCopyInF32(
                    dkWs[rowBase],
                    wsF32_[f32Base + SlotLayoutF32::dkPartialWs + static_cast<uint64_t>(r0) * bk], nElem);
                PpAcquire(sQ);
                const uint32_t sGk = PpCopyInF32(
                    gkWs[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk],
                    nElem);
                PpAcquire(sKstr);
                const uint32_t sDq = PpCopyInF32(
                    dqWs[rowBase],
                    wsF32_[f32Base + SlotLayoutF32::dqGatedWs + static_cast<uint64_t>(r0) * bk], nElem);
                Cast(qFp[rowBase], qIn, RoundMode::CAST_NONE, nElem);
                Cast(kFp[rowBase], kIn, RoundMode::CAST_NONE, nElem);
                PpAcquire(sDk);
                PpAcquire(sGk);
                PpAcquire(sDq);
                PipeBarrier<PIPE_V>();
            }
            if (validOwned < nr) {
                const uint32_t sDk = PpCopyInF32(
                    dkWs[rowBase],
                    wsF32_[f32Base + SlotLayoutF32::dkPartialWs + static_cast<uint64_t>(r0) * bk], nElem);
                const uint32_t sGk = PpCopyInF32(
                    gkWs[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk],
                    nElem);
                PpAcquire(sDk);
                const uint32_t sDq = PpCopyInF32(
                    dqWs[rowBase],
                    wsF32_[f32Base + SlotLayoutF32::dqGatedWs + static_cast<uint64_t>(r0) * bk], nElem);
                Cast(qFp[rowBase], qIn, RoundMode::CAST_NONE, nElem);
                Cast(kFp[rowBase], kIn, RoundMode::CAST_NONE, nElem);
                PpAcquire(sGk);
                PpAcquire(sDq);
                PipeBarrier<PIPE_V>();
            }
#else
            DataCopy(dkgb[rowBase], wsF32_[f32Base + SlotLayoutF32::dkgbWs + static_cast<uint64_t>(r0) * bk],
                     nElem);
            LocalTensor<T> kgIn = arenaT_.Get<T>();
            DataCopy(kgIn, wsT_[tBase + SlotLayoutT::kgWs + static_cast<uint64_t>(r0) * bk], nElem);
            SyncMte2ToV();
            Cast(kgFp[rowBase], kgIn, RoundMode::CAST_NONE, nElem);
            PipeBarrier<PIPE_V>();

            LocalTensor<T> qIn = arenaT_.Get<T>()[btbk];
            LocalTensor<T> kIn = arenaT_.Get<T>()[2 * btbk];
            if (validOwned > 0) {
                CopyStrided(qIn, q_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
                CopyStrided(kIn, k_, this->QkOff(bIdx, iH, tok0 + r0, kOff), kDim_, validOwned, bk);
            }
            if (validOwned < nr) {
                Duplicate(qIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
                Duplicate(kIn[validOwned * bk], static_cast<T>(0), (nr - validOwned) * bk);
                SetFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
                WaitFlag<HardEvent::V_MTE2>(EVT_V_MTE2);
            }
            DataCopy(dkWs[rowBase],
                     wsF32_[f32Base + SlotLayoutF32::dkPartialWs + static_cast<uint64_t>(r0) * bk], nElem);
            DataCopy(gkWs[rowBase], wsF32_[f32Base + SlotLayoutF32::gkWs + static_cast<uint64_t>(r0) * bk],
                     nElem);
            DataCopy(dqWs[rowBase],
                     wsF32_[f32Base + SlotLayoutF32::dqGatedWs + static_cast<uint64_t>(r0) * bk], nElem);
            SyncMte2ToV();
            Cast(qFp[rowBase], qIn, RoundMode::CAST_NONE, nElem);
            Cast(kFp[rowBase], kIn, RoundMode::CAST_NONE, nElem);
            PipeBarrier<PIPE_V>();
#endif
        }

        LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
        {
            const uint32_t sDb = PpCopyInF32(dbAcc, wsF32_[f32Base + DbMergeOff()], static_cast<uint32_t>(bt_));
            PpAcquire(sDb);
        }
#else
        DataCopy(dbAcc, wsF32_[f32Base + DbMergeOff()], static_cast<uint32_t>(bt_));
        SyncMte2ToV();
#endif
        if (nr > 0) {
            LocalTensor<float> scratchOwned = scratch[rowBase];
            LocalTensor<float> dkgbOwned = dkgb[rowBase];
            LocalTensor<float> kgOwned = kgFp[rowBase];
            Mul(scratchOwned, dkgbOwned, kgOwned, nElem);
            PipeBarrier<PIPE_V>();
#if USE_OWNED_ARENA
            // scratch is compact at rowBase; fold into owned half of dbAcc.
            Duplicate(dbAcc, 0.0f, static_cast<uint32_t>(bt_));
            PipeBarrier<PIPE_V>();
            LocalTensor<float> dbOwned = dbAcc[r0];
            RowFoldSumAddInto(dbOwned, scratchOwned, nr, bk);
#else
            RowFoldSumAddIntoOwned(dbAcc, scratch, static_cast<uint32_t>(bt_), bk);
#endif
        }
        SyncVToMte3();
        DataCopy(wsF32_[f32Base + DbMergeOff()], dbAcc, static_cast<uint32_t>(bt_));
#if USE_EPILOG_STORE_MERGE
        // Overlap dbAcc MTE3 writeback with following V (Mul/Sub); Wait before next MTE2.
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#else
        SyncMte3ToMte2();
#endif

        const uint32_t lastRow = static_cast<uint32_t>(validRows - 1);
        if (nr > 0) {
            LocalTensor<float> kOwned = kFp[rowBase];
            LocalTensor<float> qOwned = qFp[rowBase];
            LocalTensor<float> dkOwned = dkWs[rowBase];
            LocalTensor<float> dqOwned = dqWs[rowBase];
            LocalTensor<float> kgOwned = kgFp[rowBase];
            LocalTensor<float> dkgbOwned = dkgb[rowBase];
            LocalTensor<float> gkOwned = gkWs[rowBase];

            Mul(kOwned, kOwned, dkOwned, nElem);
#if USE_FOLD_BAR_SLIM
            // Independent dsts — one barrier before Sub that reads both.
            Mul(qOwned, qOwned, dqOwned, nElem);
            PipeBarrier<PIPE_V>();
#else
            PipeBarrier<PIPE_V>();
            Mul(qOwned, qOwned, dqOwned, nElem);
            PipeBarrier<PIPE_V>();
#endif
            Sub(qOwned, qOwned, kOwned, nElem);
            PipeBarrier<PIPE_V>();

#if USE_EPILOG_STORE_MERGE
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#endif
            LocalTensor<float> dgkRow = smallBuf_.Get<float>();
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sDgk = PpCopyInF32(dgkRow, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], bk);
                PpAcquire(sDgk);
            }
#else
            DataCopy(dgkRow, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], bk);
            SyncMte2ToV();
#endif
            if (OwnRow(lastRow)) {
                Add(qOwned[(lastRow - r0) * bk], qOwned[(lastRow - r0) * bk], dgkRow, bk);
                PipeBarrier<PIPE_V>();
            }

            Mul(kOwned, kgOwned, dkgbOwned, nElem);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> beta = betaBuf_.Get<float>();
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sBeta =
                    PpCopyInF32(beta, wsF32_[f32Base + SlotLayoutF32::betaWs], static_cast<uint32_t>(bt_));
                PpAcquire(sBeta);
            }
#else
            DataCopy(beta, wsF32_[f32Base + SlotLayoutF32::betaWs], static_cast<uint32_t>(bt_));
            SyncMte2ToV();
#endif
            LocalTensor<float> betaOwned = beta[r0];
            LocalTensor<float> brcbScratch = brcbBuf_.Get<float>();
            RowBroadcastMulInPlace(kOwned, betaOwned, brcbScratch, nr, bk);
            Add(qOwned, qOwned, kOwned, nElem);
            PipeBarrier<PIPE_V>();

            RowBroadcastMulInPlace(gkOwned, betaOwned, brcbScratch, nr, bk);
            Mul(dkgbOwned, dkgbOwned, gkOwned, nElem);
            PipeBarrier<PIPE_V>();
            Add(dkOwned, dkOwned, dkgbOwned, nElem);
#if USE_EPILOG_STORE_MERGE
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
#else
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
#endif
            CopyStridedOutOwned(dk_, this->HvKOff(bIdx, iHv, tok0, kOff), kDim_, dkWs,
                                static_cast<uint32_t>(validRows), bk, /*compactSrc=*/true);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        } else {
#if USE_EPILOG_STORE_MERGE
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#endif
            LocalTensor<float> dgkRow = smallBuf_.Get<float>();
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sDgk = PpCopyInF32(dgkRow, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], bk);
                PpAcquire(sDgk);
            }
#else
            DataCopy(dgkRow, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs + MAX_BK], bk);
            SyncMte2ToV();
#endif
        }

#if USE_OWNED_ARENA
        // Park dg (qFp): MASK dA + state VEC_FOLD reuse arena and would clobber it.
        // dkPartialWs is free after GM dk store.
        if (nr > 0) {
            SyncVToMte3();
            CopyWsRowsOwned(wsF32_, f32Base + SlotLayoutF32::dkPartialWs, qFp, static_cast<uint32_t>(bt_), bk,
                            bk);
            SyncMte3ToMte2();
        }
#endif

#if USE_MASK_SOFT_LEAD
        // Fused-only: Mask belongs to OpC under stage split (no CrossCore across launches).
        if (stageId_ == 0) {
        // Last BK: finish dA+=delta + Mask/Set before heavy state, so Cube Stage3
        // overlaps state/dg (PR190-style Set-before-tail). Safe UB past qFp slot.
        const bool lastBk = (iK + 1U == this->NumBk());
        if (lastBk && IsSub0()) {
            // Reuse arena base: owned panels no longer needed after dk/dq store.
            // (kDaTmp=4*BT*BK overflows ARENA when USE_OWNED_ARENA / BK128.)
            LocalTensor<float> dAWs = arena;
            const uint32_t btbt = static_cast<uint32_t>(bt_ * bt_);
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sDa = PpCopyInF32(dAWs, wsF32_[f32Base + SlotLayoutF32::dAWs], btbt);
                LocalTensor<float> delta = arena[btbt];
                const uint32_t sDel = PpCopyInF32(delta, wsF32_[f32Base + SlotLayoutF32::dADeltaWs], btbt);
                PpAcquire(sDa);
                PpAcquire(sDel);
                Add(dAWs, dAWs, delta, btbt);
            }
#else
            DataCopy(dAWs, wsF32_[f32Base + SlotLayoutF32::dAWs], btbt);
            LocalTensor<float> delta = arena[btbt];
            DataCopy(delta, wsF32_[f32Base + SlotLayoutF32::dADeltaWs], btbt);
            SyncMte2ToV();
            Add(dAWs, dAWs, delta, btbt);
#endif
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dAWs], dAWs, btbt);
            SyncMte3ToMte2();
        }
        if (lastBk) {
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            PipeBarrier<PIPE_MTE3>();
            Stage3MaskVec(slot, bIdx, iHv, validRows);
            SetVMaskJoined();
        }
        }
#endif

        // State + dg store. With MASK_SOFT_LEAD, this overlaps Cube Stage3 AA.
        if (OwnRow(lastRow)) {
            LocalTensor<float> gnExp = smallBuf_.Get<float>()[MAX_BK];
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sGn = PpCopyInF32(gnExp, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs], bk);
                PpAcquire(sGn);
            }
#else
            DataCopy(gnExp, wsF32_[f32Base + SlotLayoutF32::dgkMergeWs], bk);
            SyncMte2ToV();
#endif
#if !USE_EXP_GN_PARK
            Exp2InPlace(gnExp, bk);
#endif
            LocalTensor<float> stateAcc = dbAccBuf_.Get<float>();
            Duplicate(stateAcc, 0.0f, bk);
            PipeBarrier<PIPE_V>();
            const uint64_t stateBase = this->StateOff(bIdx, iHv, localChunk);
            LocalTensor<T> tIn = arenaT_.Get<T>();
            for (uint32_t iv = 0; iv < nBv; ++iv) {
                const uint32_t bv = this->BvSize(iv);
                const uint64_t vOff = static_cast<uint64_t>(iv) * MAX_BV;
                if (stateVFirst_) {
                    const uint32_t bvbk = bv * bk;
                    LocalTensor<T> hIn = tIn;
                    LocalTensor<T> dhIn2 = tIn[bvbk];
                    CopyStrided(hIn, h_, stateBase + vOff * kDim_ + kOff, kDim_, bv, bk);
                    CopyStrided(dhIn2, dhIn_, stateBase + vOff * kDim_ + kOff, kDim_, bv, bk);
                    SetFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                    WaitFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                    LocalTensor<float> hFp = scratch;
                    LocalTensor<float> dhFp = scratch[bvbk];
                    Cast(hFp, hIn, RoundMode::CAST_NONE, bvbk);
                    Cast(dhFp, dhIn2, RoundMode::CAST_NONE, bvbk);
                    PipeBarrier<PIPE_V>();
                    Mul(hFp, hFp, dhFp, bvbk);
                    PipeBarrier<PIPE_V>();
                    ColSumAddInto(stateAcc, hFp, bv, bk);
#if USE_FOLD_BAR_SLIM
                    PipeBarrier<PIPE_V>();
#endif
                } else {
#if USE_EPILOG_VEC_FOLD
                    // !stateVFirst [K,V]: one strided panel load [bk,bv] + RowFoldSum (not 64× row GM).
                    const uint32_t bkbv = bk * bv;
                    LocalTensor<T> hPanel = tIn;
                    LocalTensor<T> dhPanel = tIn[bkbv];
                    const uint64_t panelBase = stateBase + kOff * vDim_ + vOff;
                    CopyStrided(hPanel, h_, panelBase, vDim_, bk, bv);
                    CopyStrided(dhPanel, dhIn_, panelBase, vDim_, bk, bv);
                    SetFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                    WaitFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                    LocalTensor<float> hFp = arenaF32_.Get<float>();
                    LocalTensor<float> dhFp = hFp[bkbv];
                    Cast(hFp, hPanel, RoundMode::CAST_NONE, bkbv);
                    Cast(dhFp, dhPanel, RoundMode::CAST_NONE, bkbv);
                    PipeBarrier<PIPE_V>();
                    Mul(hFp, hFp, dhFp, bkbv);
                    PipeBarrier<PIPE_V>();
                    RowFoldSumAddInto(stateAcc, hFp, bk, bv);
#else
                    LocalTensor<T> hRowT = tIn;
                    LocalTensor<T> dhRowT = tIn[MAX_BV];
                    LocalTensor<float> hRow = brcbBuf_.Get<float>();
                    LocalTensor<float> dhRow = brcbBuf_.Get<float>()[MAX_BV];
                    for (uint32_t r = 0; r < bk; ++r) {
                        const uint64_t rowBaseGm =
                            stateBase + (kOff + static_cast<uint64_t>(r)) * vDim_ + vOff;
                        this->CopyVectorIn(hRowT, h_, rowBaseGm, bv);
                        this->CopyVectorIn(dhRowT, dhIn_, rowBaseGm, bv);
                        SetFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                        WaitFlag<HardEvent::MTE2_V>(EVT_STATE_MTE2_V);
                        Cast(hRow, hRowT, RoundMode::CAST_NONE, bv);
                        Cast(dhRow, dhRowT, RoundMode::CAST_NONE, bv);
                        PipeBarrier<PIPE_V>();
                        Mul(hRow, hRow, dhRow, bv);
                        PipeBarrier<PIPE_V>();
                        uint32_t remain = bv;
                        while (remain > 8) {
                            const uint32_t calcCnt = remain / 2;
                            const uint32_t newRemain = remain - calcCnt;
                            Add(hRow, hRow, hRow[newRemain], calcCnt);
                            PipeBarrier<PIPE_V>();
                            remain = newRemain;
                        }
                        SetFlag<HardEvent::V_S>(EVT_STATE_VS);
                        WaitFlag<HardEvent::V_S>(EVT_STATE_VS);
                        float sum = stateAcc.GetValue(r);
                        for (uint32_t c = 0; c < remain; ++c) {
                            sum += hRow.GetValue(c);
                        }
                        stateAcc.SetValue(r, sum);
                        SetFlag<HardEvent::S_V>(EVT_STATE_SV);
                        WaitFlag<HardEvent::S_V>(EVT_STATE_SV);
                    }
#endif
                }
            }
            Mul(stateAcc, stateAcc, gnExp, bk);
            PipeBarrier<PIPE_V>();
#if USE_OWNED_ARENA
            // Reload parked dg into compact qFp, then add state into local last row.
            DataCopy(qFp[rowBase],
                     wsF32_[f32Base + SlotLayoutF32::dkPartialWs + static_cast<uint64_t>(r0) * bk], nElem);
            SyncMte2ToV();
            Add(qFp[rowBase + (lastRow - r0) * bk], qFp[rowBase + (lastRow - r0) * bk], stateAcc, bk);
#else
            Add(qFp[lastRow * bk], qFp[lastRow * bk], stateAcc, bk);
#endif
#if !USE_EPILOG_STORE_MERGE
            PipeBarrier<PIPE_V>();
#endif
        }
#if USE_OWNED_ARENA
        else if (nr > 0) {
            // Non-last-row AIV: reload parked dg for store (arena may be clobbered by Sub0 dA).
            DataCopy(qFp[rowBase],
                     wsF32_[f32Base + SlotLayoutF32::dkPartialWs + static_cast<uint64_t>(r0) * bk], nElem);
            SyncMte2ToV();
        }
#endif
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyStridedOutOwned(dg_, this->HvKOff(bIdx, iHv, tok0, kOff), kDim_, qFp,
                            static_cast<uint32_t>(validRows), bk, /*compactSrc=*/true);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);

#if USE_MASK_SOFT_LEAD
        // Fused soft-lead: Mask already did last-BK accum. Stage split: always accum here.
        if ((stageId_ != 0 || iK + 1U < this->NumBk()) && IsSub0()) {
#else
        // dA accumulate after dg store — reuses arena (would clobber qFp if earlier).
        if (IsSub0()) {
#endif
            LocalTensor<float> dAWs = arenaF32_.Get<float>();
            const uint32_t btbt = static_cast<uint32_t>(bt_ * bt_);
#if USE_VEC_MTE2_PP && USE_VEC_MTE2_PP_EPILOG
            {
                const uint32_t sDa = PpCopyInF32(dAWs, wsF32_[f32Base + SlotLayoutF32::dAWs], btbt);
                LocalTensor<float> delta = arenaF32_.Get<float>()[btbt];
                const uint32_t sDel = PpCopyInF32(delta, wsF32_[f32Base + SlotLayoutF32::dADeltaWs], btbt);
                PpAcquire(sDa);
                PpAcquire(sDel);
                Add(dAWs, dAWs, delta, btbt);
            }
#else
            DataCopy(dAWs, wsF32_[f32Base + SlotLayoutF32::dAWs], btbt);
            LocalTensor<float> delta = arenaF32_.Get<float>()[btbt];
            DataCopy(delta, wsF32_[f32Base + SlotLayoutF32::dADeltaWs], btbt);
            SyncMte2ToV();
            Add(dAWs, dAWs, delta, btbt);
#endif
            PipeBarrier<PIPE_V>();
            SyncVToMte3();
            DataCopy(wsF32_[f32Base + SlotLayoutF32::dAWs], dAWs, btbt);
            SyncMte3ToMte2();
        }

#if USE_MERGE_BARRIER_ONLY && USE_MASK_SOFT_LEAD
        // Last BK already Join'd before Stage3MaskVec; skip duplicate trailing Join.
        if (stageId_ != 0 || iK + 1U < this->NumBk()) {
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            PipeBarrier<PIPE_MTE3>();
        }
#else
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
#endif
    }

    // ==================== Stage3: mask, AA, negate+store ====================
    // Packed bit-mask (matches the proven pattern in
    // prepare_wy_repr_bwd_da_vector.h): bit=1 marks positions to be ZEROED
    // (upper triangle j>=i, or any row >= validRows), bit=0 marks kept (j<i<validRows).
    __aicore__ inline void BuildStrictLowerMaskPacked(LocalTensor<uint8_t> &mask, uint32_t validRows)
    {
        const uint32_t rows = static_cast<uint32_t>(bt_);
        const uint32_t numBlocksPerRow = (rows + 7) / 8;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t block = 0; block < numBlocksPerRow; ++block) {
                const uint32_t colStart = block * 8;
                uint8_t maskVal = 0;
                for (uint32_t bit = 0; bit < 8 && (colStart + bit) < rows; ++bit) {
                    const uint32_t col = colStart + bit;
                    if (col >= row || row >= validRows) {
                        maskVal |= static_cast<uint8_t>(1u << bit);
                    }
                }
                mask.SetValue(row * numBlocksPerRow + block, maskVal);
            }
        }
    }

    __aicore__ inline LocalTensor<uint8_t> EnsureMask(uint32_t validRows)
    {
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        if (cachedMaskValidRows_ != validRows) {
            BuildStrictLowerMaskPacked(mask, validRows);
            SetFlag<HardEvent::S_V>(EVT_SCALAR_SV);
            WaitFlag<HardEvent::S_V>(EVT_SCALAR_SV);
            cachedMaskValidRows_ = validRows;
        }
        return mask;
    }

    __aicore__ inline void Stage3MaskVec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t validRows)
    {
        (void)bIdx;
        (void)iHv;
#if !USE_DUAL_AIV_MASK
        // Full BT×BT mask is single-writer: AIV1 only Sets V_MASK (no duplicate work).
        if (!IsSub0()) {
            return;
        }
#endif
        if (!IsComputeAiv()) {
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const uint64_t tBase = slot * SlotLayoutT::TOTAL;
        const uint32_t btbt = static_cast<uint32_t>(bt_ * bt_);

#if USE_DUAL_AIV_MASK
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 + nr > static_cast<uint32_t>(bt_)) {
            nr = static_cast<uint32_t>(bt_) - r0;
        }
        if (nr == 0) {
            return;
        }
        const uint32_t nElem = nr * static_cast<uint32_t>(bt_);
        LocalTensor<float> dA = arenaF32_.Get<float>();
        DataCopy(dA[r0 * bt_], wsF32_[f32Base + SlotLayoutF32::dAWs + static_cast<uint64_t>(r0) * bt_], nElem);
        LocalTensor<float> beta = betaBuf_.Get<float>();
        DataCopy(beta, wsF32_[f32Base + SlotLayoutF32::betaWs], static_cast<uint32_t>(bt_));
        SyncMte2ToV();

        LocalTensor<float> dAOwned = dA[r0 * bt_];
        ColBroadcastMulInPlace(dAOwned, beta, nr, static_cast<uint32_t>(bt_));

        LocalTensor<uint8_t> mask = EnsureMask(static_cast<uint32_t>(validRows));
#if USE_MASK_SELECT_SLIM
        // ColBroadcastMulInPlace already PipeBarrier'd; reuse Init zero (no Duplicate).
        LocalTensor<float> zero = zeroSelBuf_.Get<float>();
#else
        LocalTensor<float> zero = arenaF32_.Get<float>()[btbt];
        Duplicate(zero, 0.0f, 8);
        PipeBarrier<PIPE_V>();
#endif
        const uint8_t rowBlk = static_cast<uint8_t>((bt_ * sizeof(float)) / 32);
        BinaryRepeatParams repeatParams{1, 0, 1, rowBlk, 0, rowBlk};
        const uint32_t maskBytesPerRow = (static_cast<uint32_t>(bt_) + 7) / 8;
        Select(dAOwned, mask[r0 * maskBytesPerRow], zero, dAOwned, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(bt_), static_cast<uint8_t>(nr), repeatParams);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> dAT = arenaT_.Get<T>();
        LocalTensor<T> dATOwned = dAT[r0 * bt_];
        Cast(dATOwned, dAOwned, RoundMode::CAST_RINT, nElem);
#if !USE_MASK_SELECT_SLIM
        PipeBarrier<PIPE_V>();
#endif
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        DataCopy(wsT_[tBase + SlotLayoutT::dAMaskedWs + static_cast<uint64_t>(r0) * bt_], dATOwned, nElem);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#else
        LocalTensor<float> dA = arenaF32_.Get<float>();
        DataCopy(dA, wsF32_[f32Base + SlotLayoutF32::dAWs], btbt);
        LocalTensor<float> beta = betaBuf_.Get<float>();
        DataCopy(beta, wsF32_[f32Base + SlotLayoutF32::betaWs], static_cast<uint32_t>(bt_));
        SyncMte2ToV();

        ColBroadcastMulInPlace(dA, beta, static_cast<uint32_t>(bt_), static_cast<uint32_t>(bt_));

        LocalTensor<uint8_t> mask = EnsureMask(static_cast<uint32_t>(validRows));

#if USE_MASK_SELECT_SLIM
        LocalTensor<float> zero = zeroSelBuf_.Get<float>();
#else
        LocalTensor<float> zero = arenaF32_.Get<float>()[btbt];
        Duplicate(zero, 0.0f, 8);
        PipeBarrier<PIPE_V>();
#endif
        const uint8_t rowBlk = static_cast<uint8_t>((bt_ * sizeof(float)) / 32);
        BinaryRepeatParams repeatParams{1, 0, 1, rowBlk, 0, rowBlk};
        Select(dA, mask, zero, dA, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(bt_),
               static_cast<uint8_t>(bt_), repeatParams);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> dAT = arenaT_.Get<T>();
        Cast(dAT, dA, RoundMode::CAST_RINT, btbt);
#if !USE_MASK_SELECT_SLIM
        PipeBarrier<PIPE_V>();
#endif
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        DataCopy(wsT_[tBase + SlotLayoutT::dAMaskedWs], dAT, btbt);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#endif
    }

    __aicore__ inline void Stage3StoreVec(uint64_t slot, uint64_t bIdx, uint64_t iHv, uint64_t tok0,
                                          uint64_t validRows)
    {
        if (!IsComputeAiv()) {
            JoinAivBarrier();
            return;
        }
        const uint64_t f32Base = slot * SlotLayoutF32::TOTAL;
        const uint32_t btbt = static_cast<uint32_t>(bt_ * bt_);

#if USE_DUAL_AIV_STORE
        // PR190-style: owned-row Select/store; Sub0-only db merge.
        const uint32_t r0 = OwnedRowBegin();
        uint32_t nr = OwnedRowCount();
        if (r0 + nr > static_cast<uint32_t>(bt_)) {
            nr = static_cast<uint32_t>(bt_) - r0;
        }
        if (nr > 0) {
            const uint32_t nElem = nr * static_cast<uint32_t>(bt_);
            LocalTensor<float> dA3 = arenaF32_.Get<float>();
            DataCopy(dA3[r0 * bt_], wsF32_[f32Base + SlotLayoutF32::dA3Ws + static_cast<uint64_t>(r0) * bt_],
                     nElem);
            SyncMte2ToV();
            LocalTensor<float> dA3Owned = dA3[r0 * bt_];
            Muls(dA3Owned, dA3Owned, -1.0f, nElem);
            PipeBarrier<PIPE_V>();
#if !USE_MASK_ONCE
            LocalTensor<uint8_t> mask = EnsureMask(static_cast<uint32_t>(validRows));
#if USE_MASK_SELECT_SLIM
            LocalTensor<float> zero = zeroSelBuf_.Get<float>();
#else
            LocalTensor<float> zero = arenaF32_.Get<float>()[btbt];
            Duplicate(zero, 0.0f, 8);
            PipeBarrier<PIPE_V>();
#endif
            const uint8_t rowBlk = static_cast<uint8_t>((bt_ * sizeof(float)) / 32);
            BinaryRepeatParams repeatParams{1, 0, 1, rowBlk, 0, rowBlk};
            const uint32_t maskBytesPerRow = (static_cast<uint32_t>(bt_) + 7) / 8;
            Select(dA3Owned, mask[r0 * maskBytesPerRow], zero, dA3Owned, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(bt_), static_cast<uint8_t>(nr), repeatParams);
            PipeBarrier<PIPE_V>();
#endif

            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyStridedOutOwned(dAOut_, this->AOff(bIdx, iHv, tok0, 0), bt_, dA3, static_cast<uint32_t>(validRows),
                                static_cast<uint32_t>(bt_));
        }
        MergeDbAcc(slot);
        if (IsSub0()) {
            LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
            this->CopyVectorOut(db_, this->BetaOff(bIdx, iHv, tok0), dbAcc, validRows);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
#else
        LocalTensor<float> dA3 = arenaF32_.Get<float>();
        DataCopy(dA3, wsF32_[f32Base + SlotLayoutF32::dA3Ws], btbt);
        SyncMte2ToV();
        Muls(dA3, dA3, -1.0f, btbt);
        PipeBarrier<PIPE_V>();
#if !USE_MASK_ONCE
        LocalTensor<uint8_t> mask = EnsureMask(static_cast<uint32_t>(validRows));

#if USE_MASK_SELECT_SLIM
        LocalTensor<float> zero = zeroSelBuf_.Get<float>();
#else
        LocalTensor<float> zero = arenaF32_.Get<float>()[btbt];
        Duplicate(zero, 0.0f, 8);
        PipeBarrier<PIPE_V>();
#endif
        const uint8_t rowBlk = static_cast<uint8_t>((bt_ * sizeof(float)) / 32);
        BinaryRepeatParams repeatParams{1, 0, 1, rowBlk, 0, rowBlk};
        Select(dA3, mask, zero, dA3, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(bt_),
               static_cast<uint8_t>(bt_), repeatParams);
        PipeBarrier<PIPE_V>();
#endif

        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyStridedOutOwned(dAOut_, this->AOff(bIdx, iHv, tok0, 0), bt_, dA3, static_cast<uint32_t>(validRows),
                            static_cast<uint32_t>(bt_));

        MergeDbAcc(slot);
        if (IsSub0()) {
            LocalTensor<float> dbAcc = dbAccBuf_.Get<float>();
            this->CopyVectorOut(db_, this->BetaOff(bIdx, iHv, tok0), dbAcc, validRows);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
#endif
    }

    TPipe *pipe_ = nullptr;
    TBuf<> arenaF32_, arenaT_, betaBuf_, dbAccBuf_, smallBuf_, brcbBuf_, maskBuf_;
#if USE_MASK_SELECT_SLIM
    TBuf<> zeroSelBuf_;
#endif
    uint32_t subBlockNum_ = 1;
    uint32_t subBlockIdx_ = 0;
    uint32_t cachedMaskValidRows_ = 0xFFFFFFFFu;
#if USE_VEC_MTE2_PP
    event_t mte2ToVPp_[2]{};
    event_t vToMte2Pp_[2]{};
    uint32_t ppIdx_ = 0;
    bool ppInited_ = false;
#endif
    Catlass::Arch::CrossCoreFlag cS0_{FLAG_C_S0};
    Catlass::Arch::CrossCoreFlag cS1_{FLAG_C_S1};
    Catlass::Arch::CrossCoreFlag vGate_{FLAG_V_GATE};
    Catlass::Arch::CrossCoreFlag vS0_{FLAG_V_S0};
    Catlass::Arch::CrossCoreFlag cS2_{FLAG_C_S2};
    Catlass::Arch::CrossCoreFlag vMask_{FLAG_V_MASK};
    Catlass::Arch::CrossCoreFlag cS3_{FLAG_C_S3};
    Catlass::Arch::CrossCoreFlag slotFree_[NUM_GM_SLOTS] = {FLAG_SLOT_FREE0, FLAG_SLOT_FREE1, FLAG_SLOT_FREE2,
                                                             FLAG_SLOT_FREE3};
};

} // namespace kda_wy_dqkg

#endif // CHUNK_KDA_BWD_WY_DQKG_FUSED_VECTOR_H
