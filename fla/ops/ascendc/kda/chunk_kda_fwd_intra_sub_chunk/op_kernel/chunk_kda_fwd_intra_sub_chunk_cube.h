/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — Cube (AIC) side.
 * Two score GEMMs per sub-chunk (Kg held in L1 across both), fp32 output.
 *   Aqk_raw = Qg @ Kg.T ;  Akk_raw = W(=Kgq) @ Kg.T
 * NO MCH — the (I+L)^{-1} inverse is done on Vector via Forward Substitution.
 */

#ifndef CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
#define CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H

#include "chunk_kda_fwd_intra_sub_chunk_common.h"

namespace kda_isub {

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
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
                const uint64_t slot = this->SlotOf(iSub);
                Catlass::Arch::CrossCoreWaitFlag(s0Ready_);
                // Fresh Resource+BlockMmad per sub-chunk (matches chunk_kda_fwd::ComputeRawAqkAkkCubeBlock).
                ComputeMmad(slot);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeDone_);
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
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape, ElementA,
                                                         ElementB, ElementC, void, TileCopy>;

    // Aqk_raw = Qg @ Kg^T ; Akk_raw = W @ Kg^T. ElementA/B = qk dtype, ElementC = fp32.
    __aicore__ inline void ComputeMmad(uint64_t slot)
    {
        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, kDim_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);
        Catlass::GemmCoord shape{static_cast<uint32_t>(bc_), static_cast<uint32_t>(bc_), static_cast<uint32_t>(kDim_)};

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

        blockMmad(blockQg, blockKg, blockAqk, shape);
        PipeBarrier<PIPE_ALL>();
        blockMmad(blockW, blockKg, blockAkk, shape);
        PipeBarrier<PIPE_ALL>();
    }

    Catlass::Arch::CrossCoreFlag s0Ready_{FLAG_S0_READY};
    Catlass::Arch::CrossCoreFlag cubeDone_{FLAG_CUBE_DONE};
};

} // namespace kda_isub

#endif // CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
