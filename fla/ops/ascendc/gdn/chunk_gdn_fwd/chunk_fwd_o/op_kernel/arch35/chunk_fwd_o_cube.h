/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * AIC Stage2: L0/L1 ping-pong + Fixpipe→UB (A_raw/O_s_raw FP32, design §354).
 * Cross-core synchronization follows PR404's two ordered ready chains.
 */

#ifndef CHUNK_FWD_O_ARCH35_CUBE_H
#define CHUNK_FWD_O_ARCH35_CUBE_H

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

using namespace AscendC;

class ChunkFwdOA5CubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using Element = bfloat16_t;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;

    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutCM, Element,
                                                              LayoutRM>;
    using TileCopyAV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutRM, Element, LayoutRM, Element,
                                                              LayoutRM>;
    using DirectTileCopyCC = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using DirectTileCopyRM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, Element, LayoutRM, Element, LayoutCM, float, LayoutRM>;

    static constexpr uint32_t kBt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
    static constexpr uint32_t kK = static_cast<uint32_t>(CHUNK_FWD_O_A5_K);
    static constexpr uint32_t kV = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
    static constexpr uint32_t kL0BufferCount = CHUNK_FWD_O_L0_BUFFER_COUNT;
    static constexpr uint32_t kL1StreamBankCount = CHUNK_FWD_O_STREAM_BANK_COUNT;

    static_assert(CHUNK_FWD_O_L0_A_BYTES * kL0BufferCount <= ArchTag::L0A_SIZE,
                  "Stage2 L0A ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_B_BYTES * kL0BufferCount <= ArchTag::L0B_SIZE,
                  "Stage2 L0B ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L0_C_BYTES * kL0BufferCount <= ArchTag::L0C_SIZE,
                  "Stage2 L0C ping/pong exceeds architecture limit.");
    static_assert(CHUNK_FWD_O_L1_STREAM_BANK_BYTES * kL1StreamBankCount <= ArchTag::L1_SIZE,
                  "Stage2 L1 stream banks exceed architecture limit.");
    static_assert(CHUNK_FWD_O_L1_STAGE4_END <= ArchTag::L1_SIZE,
                  "Stage4 A-prime/V resident slots exceed architecture limit.");

    // Preserve the original per-slot L1 event pairing: slot0 uses 5/6 and
    // slot1 uses 7/8. These IDs avoid Catlass' lower internal event slots.
    static constexpr TEventID kL1Mte1Mte2Base = 5;
    static constexpr TEventID kL1Mte2Mte1Base = 6;

    __aicore__ inline ChunkFwdOA5CubeProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                             GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o,
                                             GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling)
    {
        tiling_ = tiling;
        qGm_.SetGlobalBuffer((__gm__ Element *)q_);
        kGm_.SetGlobalBuffer((__gm__ Element *)k_);
        vGm_.SetGlobalBuffer((__gm__ Element *)v_);
        hGm_.SetGlobalBuffer((__gm__ Element *)h_);
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<Element>(static_cast<Element>(0));
            for (uint32_t slotIdx = 0; slotIdx < kL1StreamBankCount; ++slotIdx) {
                SetFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(slotIdx));
            }
            for (uint32_t slotIdx = 0; slotIdx < kL0BufferCount; ++slotIdx) {
                SetFlag<HardEvent::M_MTE1>(L0AEvent(slotIdx));
                SetFlag<HardEvent::M_MTE1>(L0BEvent(slotIdx));
                SetFlag<HardEvent::FIX_M>(slotIdx);
            }
            l0ASlot_ = 0U;
            l0BSlot_ = 0U;
            l0CSlot_ = 0U;
            l1StreamSlot_ = 0U;
            cachedLoopIdx_[0] = static_cast<uint32_t>(-1);
            cachedLoopIdx_[1] = static_cast<uint32_t>(-1);
            cachedHk_[0] = -1;
            cachedHk_[1] = -1;
        }
    }

    __aicore__ inline void Process(uint32_t coreIdx, uint32_t coreNum)
    {
        ChunkFwdOChunkLoc loc;
        for (uint32_t loopIdx = 0; loopIdx < static_cast<uint32_t>(tiling_.chunkNum); ++loopIdx) {
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, loopIdx, loc);
            if (coreIdx != (loopIdx % coreNum)) {
                continue;
            }
            for (int64_t hvBase = 0; hvBase < tiling_.vNumHead; hvBase += tiling_.taskGroupSize) {
                const int64_t remaining = tiling_.vNumHead - hvBase;
                const int64_t taskCount = remaining < tiling_.taskGroupSize ? remaining : tiling_.taskGroupSize;
                const int64_t groupRound =
                    ChunkFwdOGroupRound(tiling_, loopIdx, coreIdx, coreNum, hvBase);
                // Stage 2: consume the Stage 1 ready chain and produce QK/QH
                // for every HEAD in this task group before entering Stage 4.
                if ASCEND_IS_AIC {
                    l1StreamSlot_ = 0U;
                    for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                        const int64_t hv = hvBase + headOffset;
                        const int64_t hk = hv / tiling_.hvPerHk;
                        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                        const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                        const bool loadQK = cachedLoopIdx_[l1StreamSlot_] != loopIdx ||
                                            cachedHk_[l1StreamSlot_] != hk;
                        ProcessStage2Head(loc, hk, hv, ownerSubBlock, localSlot, l1StreamSlot_, loadQK);
                        cachedLoopIdx_[l1StreamSlot_] = loopIdx;
                        cachedHk_[l1StreamSlot_] = hk;
                        l1StreamSlot_ ^= 1U;
                    }
                }

                // Stage 4: consume Stage 3 A-prime ready signals, compute
                // A-prime@V, and publish O_l for every HEAD in this group.
                if ASCEND_IS_AIC {
                    l1StreamSlot_ = 0U;
                    for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                        Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
                        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                        const uint32_t localSlot = static_cast<uint32_t>(headOffset / 2);
                        ProcessStage4Head(loc, hvBase + headOffset, ownerSubBlock, localSlot,
                                          static_cast<uint32_t>(headOffset), groupRound, l1StreamSlot_);
                        l1StreamSlot_ ^= 1U;
                    }
                    // Stage4 uses [256, 352) KiB in L1, which overlaps slot1's Q/K cache.
                    cachedLoopIdx_[1] = static_cast<uint32_t>(-1);
                    cachedHk_[1] = -1;
                }
                Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
            }
        }
    }

private:
    static constexpr TEventID L0AEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot);
    }

    static constexpr TEventID L0BEvent(uint32_t slot)
    {
        return static_cast<TEventID>(2U * slot + 1U);
    }

    static constexpr TEventID L1Mte1Mte2Event(uint32_t slot)
    {
        return static_cast<TEventID>(kL1Mte1Mte2Base + slot * 2U);
    }

    static constexpr TEventID L1Mte2Mte1Event(uint32_t slot)
    {
        return static_cast<TEventID>(kL1Mte2Mte1Base + slot * 2U);
    }

    __aicore__ inline void LoadStage4Inputs(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t headOffset,
                                            int64_t groupRound, uint32_t l1StreamSlot)
    {
        const uint32_t m = kBt;
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const int64_t vOffset = ChunkFwdOVOOffset(tiling_, loc, hv);

        WaitFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));

        GlobalTensor<Element> aPrimeGm;
        aPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(
            ChunkFwdOAPrimeGmOffset(workspace_, tiling_, coreIdx, groupRound, headOffset)));
        auto layoutAPrimeGm = tla::MakeLayout<Element, LayoutRM>(m, m);
        auto layoutVGm = tla::MakeLayout<Element, LayoutRM>(m, kV);
        auto tensorAPrimeGm = tla::MakeTensor(aPrimeGm, layoutAPrimeGm, Catlass::Arch::PositionGM{});
        auto tensorVGm = tla::MakeTensor(vGm_[vOffset], layoutVGm, Catlass::Arch::PositionGM{});
        auto blockAPrime = GetTile(tensorAPrimeGm, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        auto blockV = GetTile(
            tensorVGm, tla::MakeCoord(0, 0),
            tla::MakeShape(static_cast<uint32_t>(loc.chunkLen), kV));

        using LayoutTagL1A = typename TileCopyAV::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyAV::LayoutTagL1B;
        using CopyGmToL1A = typename TileCopyAV::template CopyGmToL1A<decltype(blockAPrime)>;
        using CopyGmToL1B = typename TileCopyAV::template CopyGmToL1B<decltype(blockV)>;

        LocalTensor<Element> l1APrime =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1APrimeOffset(headOffset));
        LocalTensor<Element> l1V =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1VOffset(headOffset));
        auto layoutL1APrime = tla::MakeLayout<Element, LayoutTagL1A>(m, m);
        auto layoutL1V = tla::MakeLayout<Element, LayoutTagL1B>(m, kV);
        auto tensorL1APrime = tla::MakeTensor(l1APrime, layoutL1APrime, Catlass::Arch::PositionL1{});
        auto tensorL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});

        CopyGmToL1A{}(tensorL1APrime, blockAPrime);
        CopyGmToL1B{}(tensorL1V, blockV);
        SetFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));
    }

    __aicore__ inline void ProcessStage4Head(const ChunkFwdOChunkLoc &loc, int64_t hv,
                                             uint32_t ownerSubBlock, uint32_t localSlot,
                                             uint32_t headOffset, int64_t groupRound, uint32_t l1StreamSlot)
    {
        LoadStage4Inputs(loc, hv, headOffset, groupRound, l1StreamSlot);
        WaitFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));

        const uint32_t avASlot = l0ASlot_;
        const uint32_t avBSlot = l0BSlot_;
        const uint32_t avCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;

        const uint32_t mActual = static_cast<uint32_t>(loc.chunkLen);
        if (mActual == kBt) {
            RunGemmAVFull(headOffset, avASlot, avBSlot, avCSlot);
        } else {
            RunGemmAVTail(mActual, mActual, headOffset, avASlot, avBSlot, avCSlot);
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));
        PublishOl(mActual, ownerSubBlock, localSlot, avCSlot);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
    }

    __aicore__ inline void LoadQK(const ChunkFwdOChunkLoc &loc, int64_t hk, uint32_t l1StreamSlot)
    {
        const uint32_t m = kBt;
        const uint32_t mActual = static_cast<uint32_t>(loc.chunkLen);
        const int64_t qOffset = ChunkFwdOQKOffset(tiling_, loc, hk);
        const int64_t kOffset = ChunkFwdOQKOffset(tiling_, loc, hk);

        using LayoutTagL1Q = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1K = typename TileCopyQK::LayoutTagL1B;

        auto layoutQGm = tla::MakeLayout<Element, LayoutRM>(m, kK);
        auto layoutKGm = tla::MakeLayout<Element, LayoutCM>(kK, m);
        auto tensorQGm = tla::MakeTensor(qGm_[qOffset], layoutQGm, Catlass::Arch::PositionGM{});
        auto tensorKGm = tla::MakeTensor(kGm_[kOffset], layoutKGm, Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQGm, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kK));
        auto blockK = GetTile(tensorKGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, mActual));

        using CopyGmToL1Q = typename TileCopyQK::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1K = typename TileCopyQK::template CopyGmToL1B<decltype(blockK)>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(l1StreamSlot));
        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1Q>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1K>(kK, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});

        CopyGmToL1Q{}(tensorL1Q, blockQ);
        CopyGmToL1K{}(tensorL1K, blockK);
    }

    __aicore__ inline void LoadH(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t l1StreamSlot)
    {
        const int64_t hOffset = ChunkFwdOHOffset(tiling_, loc, hv);
        using LayoutTagL1H = typename TileCopyQH::LayoutTagL1B;
        auto layoutHGm = tla::MakeLayout<Element, LayoutCM>(kK, kV);
        auto tensorHGm = tla::MakeTensor(hGm_[hOffset], layoutHGm, Catlass::Arch::PositionGM{});
        auto blockH = GetTile(tensorHGm, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        using CopyGmToL1H = typename TileCopyQH::template CopyGmToL1B<decltype(blockH)>;

        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(l1StreamSlot));
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1H>(kK, kV);
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});
        CopyGmToL1H{}(tensorL1H, blockH);
    }

    __aicore__ inline void ProcessStage2Head(const ChunkFwdOChunkLoc &loc, int64_t hk, int64_t hv,
                                             uint32_t ownerSubBlock, uint32_t localSlot,
                                             uint32_t l1StreamSlot, bool loadQK)
    {
        const uint32_t m = kBt;

        WaitFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));
        if (loadQK) {
            LoadQK(loc, hk, l1StreamSlot);
            SetFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));
            WaitFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));
        }

        const uint32_t qktASlot = l0ASlot_;
        const uint32_t qktBSlot = l0BSlot_;
        const uint32_t qktCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        RunGemmQKT(m, l1StreamSlot, qktASlot, qktBSlot, qktCSlot);

        // Q/K have entered L0, so H can use MTE2 while QK runs on M/FIX.
        LoadH(loc, hv, l1StreamSlot);
        SetFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));
        PublishQKT(m, ownerSubBlock, localSlot, qktCSlot);
        WaitFlag<HardEvent::MTE2_MTE1>(L1Mte2Mte1Event(l1StreamSlot));

        const uint32_t qhASlot = l0ASlot_;
        const uint32_t qhBSlot = l0BSlot_;
        const uint32_t qhCSlot = l0CSlot_;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        RunGemmQH(m, l1StreamSlot, qhASlot, qhBSlot, qhCSlot);
        PublishQH(m, ownerSubBlock, localSlot, qhCSlot);

        SetFlag<HardEvent::MTE1_MTE2>(L1Mte1Mte2Event(l1StreamSlot));

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
    }

    template <typename DirectTileCopy, typename TensorL0C>
    __aicore__ inline void PublishDirectTile(TensorL0C &tensorL0C, uint32_t m, uint32_t n, uint32_t ubOffset,
                                             uint32_t ownerSubBlock, uint32_t l0CSlot)
    {
        auto layoutUb = tla::MakeLayout<float, LayoutRM>(m, n);
        auto tensorUb = tla::MakeTensor(resource_.ubBuf.template GetBufferByByte<float>(ubOffset), layoutUb,
                                        Catlass::Arch::PositionUB{});
        using CopyL0CToDst = typename DirectTileCopy::template CopyL0CToDst<decltype(tensorUb)>;
        CopyL0CToDst copyL0CToDst;

        SetFlag<HardEvent::M_FIX>(l0CSlot);
        WaitFlag<HardEvent::M_FIX>(l0CSlot);
        copyL0CToDst(tensorUb, tensorL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        SetFlag<HardEvent::FIX_M>(l0CSlot);
    }

    __aicore__ inline void PublishQKT(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot, uint32_t l0CSlot)
    {
        LocalTensor<float> qktL0C =
            resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));
        auto qktLayout = tla::MakeLayoutL0C(m, m);
        auto qktTensor = tla::MakeTensor(qktL0C, qktLayout, Catlass::Arch::PositionL0C{});
        auto qktTile = GetTile(qktTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, m));
        PublishDirectTile<DirectTileCopyCC>(qktTile, m, m, ChunkFwdOARawOffset(localSlot), ownerSubBlock, l0CSlot);
    }

    __aicore__ inline void PublishQH(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot, uint32_t l0CSlot)
    {
        LocalTensor<float> qhL0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));
        auto qhLayout = tla::MakeLayoutL0C(m, kV);
        auto qhTensor = tla::MakeTensor(qhL0C, qhLayout, Catlass::Arch::PositionL0C{});
        auto qhTile = GetTile(qhTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        PublishDirectTile<DirectTileCopyRM>(qhTile, m, kV, ChunkFwdOOSRawOffset(localSlot), ownerSubBlock, l0CSlot);
    }

    __aicore__ inline void PublishOl(uint32_t m, uint32_t ownerSubBlock, uint32_t localSlot, uint32_t l0CSlot)
    {
        LocalTensor<float> olL0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));
        auto olLayout = tla::MakeLayoutL0C(m, kV);
        auto olTensor = tla::MakeTensor(olL0C, olLayout, Catlass::Arch::PositionL0C{});
        auto olTile = GetTile(olTensor, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));
        PublishDirectTile<DirectTileCopyRM>(olTile, m, kV, ChunkFwdOOlOffset(localSlot), ownerSubBlock, l0CSlot);
    }

    __aicore__ inline void RunGemmQKT(uint32_t m, uint32_t l1StreamSlot, uint32_t l0ASlot, uint32_t l0BSlot,
                                      uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyQK::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQK::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQK::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQK::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQK::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQK::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1K = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1KOffset(l1StreamSlot));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1A>(m, kK);
        auto layoutL1K = tla::MakeLayout<Element, LayoutTagL1B>(kK, m);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(m, kK);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kK, m);
        auto layoutL0C = tla::MakeLayoutL0C(m, m);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(l1K, layoutL1K, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1Q = GetTile(tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL1K = GetTile(tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, m));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, m));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1Q);
        CopyL1ToL0B{}(tileL0B, tileL1K);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tileL0A, tileL0B, m, m, kK, true, 0);
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
    }

    __aicore__ inline void RunGemmQH(uint32_t m, uint32_t l1StreamSlot, uint32_t l0ASlot, uint32_t l0BSlot,
                                     uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyQH::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyQH::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyQH::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyQH::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyQH::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyQH::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1Q = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1QOffset(l1StreamSlot));
        LocalTensor<Element> l1H = resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1HOffset(l1StreamSlot));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1Q = tla::MakeLayout<Element, LayoutTagL1A>(m, kK);
        auto layoutL1H = tla::MakeLayout<Element, LayoutTagL1B>(kK, kV);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(m, kK);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kK, kV);
        auto layoutL0C = tla::MakeLayoutL0C(m, kV);
        auto tensorL1Q = tla::MakeTensor(l1Q, layoutL1Q, Catlass::Arch::PositionL1{});
        auto tensorL1H = tla::MakeTensor(l1H, layoutL1H, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1Q = GetTile(tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL1H = GetTile(tensorL1H, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, kK));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kK, kV));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, kV));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1Q);
        CopyL1ToL0B{}(tileL0B, tileL1H);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tileL0A, tileL0B, m, kV, kK, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
    }

    __aicore__ inline void RunGemmAVFull(uint32_t headOffset, uint32_t l0ASlot, uint32_t l0BSlot,
                                         uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyAV::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyAV::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyAV::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyAV::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyAV::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyAV::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1APrime =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1APrimeOffset(headOffset));
        LocalTensor<Element> l1V =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1VOffset(headOffset));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1APrime = tla::MakeLayout<Element, LayoutTagL1A>(kBt, kBt);
        auto layoutL1V = tla::MakeLayout<Element, LayoutTagL1B>(kBt, kV);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(kBt, kBt);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kBt, kV);
        auto layoutL0C = tla::MakeLayoutL0C(kBt, kV);
        auto tensorL1APrime = tla::MakeTensor(l1APrime, layoutL1APrime, Catlass::Arch::PositionL1{});
        auto tensorL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1APrime = GetTile(tensorL1APrime, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kBt));
        auto tileL1V = GetTile(tensorL1V, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kBt));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(kBt, kV));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1APrime);
        CopyL1ToL0B{}(tileL0B, tileL1V);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tileL0A, tileL0B, kBt, kV, kBt, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
    }

    __aicore__ inline void RunGemmAVTail(uint32_t mActual, uint32_t kActual, uint32_t headOffset,
                                         uint32_t l0ASlot, uint32_t l0BSlot, uint32_t l0CSlot)
    {
        using LayoutTagL1A = typename TileCopyAV::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyAV::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyAV::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyAV::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyAV::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyAV::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LayoutTagL1A>;

        LocalTensor<Element> l1APrime =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1APrimeOffset(headOffset));
        LocalTensor<Element> l1V =
            resource_.l1Buf.template GetBufferByByte<Element>(ChunkFwdOL1VOffset(headOffset));
        LocalTensor<Element> l0A =
            resource_.l0ABuf.template GetBufferByByte<Element>(ChunkFwdOL0AOffset(l0ASlot));
        LocalTensor<Element> l0B =
            resource_.l0BBuf.template GetBufferByByte<Element>(ChunkFwdOL0BOffset(l0BSlot));
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(ChunkFwdOL0COffset(l0CSlot));

        auto layoutL1APrime = tla::MakeLayout<Element, LayoutTagL1A>(kBt, kBt);
        auto layoutL1V = tla::MakeLayout<Element, LayoutTagL1B>(kBt, kV);
        auto layoutL0A = tla::MakeLayout<Element, LayoutTagL0A>(mActual, kActual);
        auto layoutL0B = tla::MakeLayout<Element, LayoutTagL0B>(kActual, kV);
        auto layoutL0C = tla::MakeLayoutL0C(mActual, kV);
        auto tensorL1APrime = tla::MakeTensor(l1APrime, layoutL1APrime, Catlass::Arch::PositionL1{});
        auto tensorL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1APrime = GetTile(tensorL1APrime, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kActual));
        auto tileL1V = GetTile(tensorL1V, tla::MakeCoord(0, 0), tla::MakeShape(kActual, kV));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kActual));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(kActual, kV));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kV));

        WaitFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        WaitFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
        WaitFlag<HardEvent::FIX_M>(l0CSlot);
        CopyL1ToL0A{}(tileL0A, tileL1APrime);
        CopyL1ToL0B{}(tileL0B, tileL1V);
        SetFlag<HardEvent::MTE1_M>(l0CSlot);
        WaitFlag<HardEvent::MTE1_M>(l0CSlot);
        TileMmad{}(tileL0C, tensorL0A, tensorL0B, true, 0);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0AEvent(l0ASlot));
        SetFlag<HardEvent::M_MTE1>(L0BEvent(l0BSlot));
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR h_;
    GM_ADDR g_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkOffsets_;
    GM_ADDR o_;
    GM_ADDR workspace_;
    ChunkFwdOTilingData tiling_{};
    Catlass::Arch::Resource<ArchTag> resource_;
    GlobalTensor<Element> qGm_;
    GlobalTensor<Element> kGm_;
    GlobalTensor<Element> vGm_;
    GlobalTensor<Element> hGm_;
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{CHUNK_FWD_O_VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CHUNK_FWD_O_CUBE_TO_VEC_READY_FLAG};
    uint32_t l0ASlot_ = 0U;
    uint32_t l0BSlot_ = 0U;
    uint32_t l0CSlot_ = 0U;
    uint32_t l1StreamSlot_ = 0U;
    uint32_t cachedLoopIdx_[kL1StreamBankCount] = {
        static_cast<uint32_t>(-1), static_cast<uint32_t>(-1)};
    int64_t cachedHk_[kL1StreamBankCount] = {-1, -1};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_CUBE_H
