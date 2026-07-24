/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — Cube (AIC) side.
 *
 * Tile-level dual score GEMM (NOT BlockMmad):
 *   Aqk_raw = Qg @ Kg.T ;  Akk_raw = W(=Kgq) @ Kg.T
 *   - Kg loaded once into L1B and held across both MMADs
 *   - W GM→L1A overlapped with MMAD1 (L1A free after L1→L0), not with Fixpipe
 *
 * NO MCH — (I+L)^{-1} is Vector Forward Substitution.
 */

#ifndef CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
#define CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H

#include "chunk_kda_fwd_intra_sub_chunk_common.h"

namespace kda_isub {

#ifndef USE_SCORE_MMAD1_LOAD_W
#define USE_SCORE_MMAD1_LOAD_W 1
#endif

template <typename T>
class KdaSubChunkCube : public KdaSubChunkBase<T> {
    using Base = KdaSubChunkBase<T>;
    using Base::bc_;
    using Base::kDim_;
    using Base::hv_;
    using Base::nc_;
    using Base::totalTasks_;
    using Base::usedCoreNum_;
    using Base::coreIdx_;
    using Base::scoreWs_;
    using Base::cmatWs_;

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR aqk, GM_ADDR akkd, GM_ADDR userWS,
                                const ChunkKdaFwdIntraSubChunkTilingData &tiling, TPipe *pipe)
    {
        (void)pipe;
        this->InitCommon(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tiling);
        coreIdx_ = static_cast<uint64_t>(GetBlockIdx());
    }

    __aicore__ inline void Process()
    {
        if (!this->ValidShapes()) {
            return;
        }
        Catlass::Arch::Resource<KdaArchTag> resource;
        uint64_t tasksOnCore = 0;
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            const uint64_t nHvWin = this->NumHvWindows();
            const uint64_t W = nc_ * nHvWin;
            if (W == 0) {
                continue;
            }
            ++tasksOnCore;
            for (uint64_t w = 0; w < W; ++w) {
                const uint64_t slot0 = this->SlotOfWindow(w, 0);
                const uint64_t slot1 = this->SlotOfWindow(w, 1);
                const uint64_t hvBase = (w % nHvWin) * 2ULL;
                // Depth: WaitS0 only. Slot reuse safe via AIV WaitCube before S0(w+2).
                Catlass::Arch::CrossCoreWaitFlag(s0Ready_);
                ComputeScoreTile(slot0, resource);
                if (hvBase + 1ULL < hv_) {
                    ComputeScoreTile(slot1, resource);
                }
                PipeBarrier<PIPE_FIX>();
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeDone_);
            }
        }
        // Match startup SetFree×4 once (Process bookend; not per-task).
        if (tasksOnCore > 0) {
            for (uint32_t s = 0; s < NUM_GM_SLOTS; ++s) {
                Catlass::Arch::CrossCoreWaitFlag(slotFree_[s]);
            }
        }
    }

private:
    using ElementA = T;
    using ElementB = T;
    using ElementC = float;
    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = Catlass::layout::ColumnMajor;
    using LayoutTagC = Catlass::layout::RowMajor;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                            ElementC, LayoutTagC>;

    // Keep clear of Catlass priming (0/1). SCORE_EVT_W separate from SCORE_EVT.
    static constexpr uint16_t SCORE_EVT = 3;
    static constexpr uint16_t SCORE_EVT_W = 4;

    __aicore__ inline void ComputeScoreTile(uint64_t slot, Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        const uint32_t m = static_cast<uint32_t>(bc_);
        const uint32_t n = static_cast<uint32_t>(bc_);
        const uint32_t k = static_cast<uint32_t>(kDim_);
        Catlass::GemmCoord shape{m, n, k};

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, kDim_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);

        auto tensorQg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_QG, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_W, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorKg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_KG, 0, 0)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorAqk =
            tla::MakeTensor(cmatWs_[this->CmatOff(slot, PLANE_AQK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk =
            tla::MakeTensor(cmatWs_[this->CmatOff(slot, PLANE_AKK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});

        auto blockQg = GetTile(tensorQg, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));

        const uint32_t l1ABytes = m * k * sizeof(ElementA);
        LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(l1ABytes);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);

        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQg)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKg)>;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(blockAqk)>;
#else
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<decltype(blockAqk)>;
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

        // --- MMAD1: Qg @ Kg → Aqk_raw；Kg→L1B once ---
        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyGmToL1A(tL1A, blockQg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);

#if USE_SCORE_MMAD1_LOAD_W
        // L1A free (Qg already in L0A); MTE2(W) ‖ MMAD1. Do NOT overlap with Fixpipe(Aqk)
        // — that path corrupted multi-HV Aqk (GM FIX vs MTE2 hazard).
        copyGmToL1A(tL1A, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT_W);
#endif
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        SetFlag<HardEvent::M_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::M_MTE1>(SCORE_EVT);
        copyL0CToGm(blockAqk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);

        // --- MMAD2: W @ Kg → Akk_raw；L1B(Kg) resident ---
#if USE_SCORE_MMAD1_LOAD_W
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT_W);
#else
        copyGmToL1A(tL1A, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
#endif
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        copyL0CToGm(blockAkk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        PipeBarrier<PIPE_ALL>();
    }

    Catlass::Arch::CrossCoreFlag s0Ready_{FLAG_S0_READY};
    Catlass::Arch::CrossCoreFlag cubeDone_{FLAG_CUBE_DONE};
    Catlass::Arch::CrossCoreFlag slotFree_[NUM_GM_SLOTS] = {FLAG_SLOT_FREE0, FLAG_SLOT_FREE1, FLAG_SLOT_FREE2,
                                                             FLAG_SLOT_FREE3};
};

} // namespace kda_isub

#endif // CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
