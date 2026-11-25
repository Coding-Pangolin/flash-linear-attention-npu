#pragma once

#include "chunk_gemm.h"
#include "stage2_prepare.h"

namespace KcpState {
using KcpPrepare::kChunk;
using KcpPrepare::kDim;
using KcpPrepare::kFeatureElements;
using KcpPrepare::kHalf;
constexpr uint32_t kStateElements = kDim * kDim;
constexpr uint32_t kBfScratchBytes = (kStateElements + 2 * kFeatureElements) * sizeof(bfloat16_t);
constexpr uint32_t kScratchBytes = kBfScratchBytes + (3 * kStateElements + 3 * kFeatureElements) * sizeof(float);

struct StateScratch {
    GM_ADDR shadow;
    GM_ADDR kgT;
    GM_ADDR innovation;
    GM_ADDR projected;
    GM_ADDR queryState;
    GM_ADDR queryLocal;
    GM_ADDR delta;
    GM_ADDR affine;
    GM_ADDR matrixNext;
};

__aicore__ inline StateScratch MakeStateScratch(GM_ADDR base)
{
    GM_ADDR bf = base + kStateElements * sizeof(bfloat16_t);
    GM_ADDR fp = base + kBfScratchBytes;
    GM_ADDR delta = fp + 3 * kFeatureElements * sizeof(float);
    return {base, bf, bf + kFeatureElements * sizeof(bfloat16_t), fp,
            fp + kFeatureElements * sizeof(float), fp + 2 * kFeatureElements * sizeof(float),
            delta, delta + kStateElements * sizeof(float), delta + 2 * kStateElements * sizeof(float)};
}

constexpr uint16_t kShadowReady = 8;
constexpr uint16_t kProjectionReady = 9;
constexpr uint16_t kInnovationReady = 10;
constexpr uint16_t kUpdateReady = 11;
constexpr uint16_t kAffineReady = 13;
constexpr uint16_t kMatrixDone = 14;
constexpr uint16_t kSecondLane = 16;

template<uint16_t Flag>
__aicore__ inline void WaitBoth()
{
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(Flag);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(Flag + kSecondLane);
}

template<uint16_t Flag>
__aicore__ inline void PublishBoth()
{
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(Flag);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(Flag + kSecondLane);
}

__aicore__ inline void RunStateCube(
    ChunkGemm& gemm, const StateScratch& tmp, GM_ADDR factors, bool forward, GM_ADDR matrix = nullptr)
{
    constexpr uint32_t kField = kFeatureElements * sizeof(bfloat16_t);
    WaitBoth<kShadowReady>();
    gemm.Run<kChunk, kDim, kDim>(factors, tmp.shadow, tmp.projected);
    if (forward) {
        gemm.Run<kChunk, kDim, kDim>(factors + 3 * kField, tmp.shadow, tmp.queryState);
    }
    PublishBoth<kProjectionReady>();
    WaitBoth<kInnovationReady>();
    gemm.Run<kDim, kDim, kChunk>(tmp.kgT, tmp.innovation, tmp.delta);
    if (forward) {
        gemm.Run<kChunk, kDim, kChunk>(factors + 4 * kField, tmp.innovation, tmp.queryLocal);
    }
    if (matrix != nullptr) {
        gemm.Run<kDim, kDim, kChunk>(tmp.kgT, factors, tmp.affine);
    }
    PublishBoth<kUpdateReady>();
    if (matrix != nullptr) {
        WaitBoth<kAffineReady>();
        gemm.Run<kDim, kDim, kDim, float>(tmp.affine, matrix, tmp.matrixNext);
        PublishBoth<kMatrixDone>();
    }
}

class StateVector {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(fp_, 3 * kDim * sizeof(float));
        pipe.InitBuffer(bf_, (kDim + kHalf * kChunk) * sizeof(bfloat16_t));
    }

    __aicore__ inline void Run(const StateScratch& tmp, GM_ADDR factors, GM_ADDR decay, GM_ADDR state,
                              GM_ADDR output, uint32_t validRows, uint32_t lane, bool forward,
                              GM_ADDR matrix = nullptr, uint32_t outputStride = kDim)
    {
        using namespace AscendC;
        GlobalTensor<bfloat16_t> prepared, shadow, kgT, innovation, out;
        GlobalTensor<float> h, d, project, delta;
        prepared.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(factors));
        shadow.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.shadow));
        kgT.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.kgT));
        innovation.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.innovation));
        h.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay));
        project.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.projected));
        delta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.delta));
        auto x = fp_.Get<float>();
        auto y = x[kDim];
        auto dRows = x[2 * kDim];
        auto bf = bf_.Get<bfloat16_t>();
        auto transposed = bf[kDim];
        const auto rowBits = bf.ReinterpretCast<uint16_t>();
        auto tileBits = transposed.ReinterpretCast<uint16_t>();
        const uint32_t firstStateRow = lane * kHalf;
        constexpr uint32_t kHalfChunk = kChunk / 2;
        const uint32_t firstToken = lane * kHalfChunk;

        DataCopy(dRows, d[firstStateRow], kHalf);
        for (uint32_t row = firstStateRow; row < firstStateRow + kHalf; ++row) {
            const uint32_t offset = row * kDim;
            DataCopy(x, h[offset], kDim);
            PipeBarrier<PIPE_ALL>();
            Cast(bf, x, RoundMode::CAST_RINT, kDim);
            PipeBarrier<PIPE_ALL>();
            DataCopy(shadow[offset], bf, kDim);
            PipeBarrier<PIPE_ALL>();
        }
        // Own K rows after transposing the chunk-major Kg operand.
        for (uint32_t t = 0; t < kChunk; ++t) {
            DataCopy(bf, prepared[2 * kFeatureElements + t * kDim + firstStateRow], kHalf);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < kHalf; ++i) {
                tileBits.SetValue(i * kChunk + t, rowBits.GetValue(i));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(kgT[firstStateRow * kChunk], transposed, kHalf * kChunk);
        CrossCoreSetFlag<4, PIPE_MTE3>(kShadowReady);
        CrossCoreWaitFlag<4, PIPE_MTE2>(kProjectionReady);
        for (uint32_t t = firstToken; t < firstToken + kHalfChunk; ++t) {
            const uint32_t offset = t * kDim;
            DataCopy(bf, prepared[kFeatureElements + offset], kDim);
            DataCopy(y, project[offset], kDim);
            PipeBarrier<PIPE_ALL>();
            Cast(x, bf, RoundMode::CAST_NONE, kDim);
            PipeBarrier<PIPE_V>();
            Sub(x, x, y, kDim);
            PipeBarrier<PIPE_V>();
            Cast(bf, x, RoundMode::CAST_RINT, kDim);
            PipeBarrier<PIPE_ALL>();
            DataCopy(innovation[offset], bf, kDim);
            PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<4, PIPE_MTE3>(kInnovationReady);
        CrossCoreWaitFlag<4, PIPE_MTE2>(kUpdateReady);
        for (uint32_t row = firstStateRow; row < firstStateRow + kHalf; ++row) {
            const uint32_t offset = row * kDim;
            DataCopy(x, h[offset], kDim);
            DataCopy(y, delta[offset], kDim);
            PipeBarrier<PIPE_ALL>();
            Muls(x, x, dRows.GetValue(row - firstStateRow), kDim);
            PipeBarrier<PIPE_V>();
            Add(x, x, y, kDim);
            PipeBarrier<PIPE_ALL>();
            DataCopy(h[offset], x, kDim);
            PipeBarrier<PIPE_ALL>();
        }
        if (forward) {
            GlobalTensor<float> fromState, fromLocal;
            fromState.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.queryState));
            fromLocal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.queryLocal));
            out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
            for (uint32_t t = firstToken; t < validRows && t < firstToken + kHalfChunk; ++t) {
                const uint32_t offset = t * kDim;
                DataCopy(x, fromState[offset], kDim);
                DataCopy(y, fromLocal[offset], kDim);
                PipeBarrier<PIPE_ALL>();
                Add(x, x, y, kDim);
                PipeBarrier<PIPE_V>();
                Cast(bf, x, RoundMode::CAST_RINT, kDim);
                PipeBarrier<PIPE_ALL>();
                DataCopy(out[t * outputStride], bf, kDim);
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (matrix != nullptr) {
            GlobalTensor<float> affine, next, current;
            affine.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.affine));
            next.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tmp.matrixNext));
            current.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
            for (uint32_t row = firstStateRow; row < firstStateRow + kHalf; ++row) {
                const uint32_t offset = row * kDim;
                DataCopy(x, affine[offset], kDim);
                PipeBarrier<PIPE_ALL>();
                Muls(x, x, -1.0f, kDim);
                PipeBarrier<PIPE_ALL>();
                x.SetValue(row, x.GetValue(row) + dRows.GetValue(row - firstStateRow));
                PipeBarrier<PIPE_ALL>();
                DataCopy(affine[offset], x, kDim);
                PipeBarrier<PIPE_ALL>();
            }
            CrossCoreSetFlag<4, PIPE_MTE3>(kAffineReady);
            CrossCoreWaitFlag<4, PIPE_MTE2>(kMatrixDone);
            for (uint32_t row = firstStateRow; row < firstStateRow + kHalf; ++row) {
                const uint32_t offset = row * kDim;
                DataCopy(x, next[offset], kDim);
                PipeBarrier<PIPE_ALL>();
                DataCopy(current[offset], x, kDim);
                PipeBarrier<PIPE_ALL>();
            }
        }
        // Do not let Cube overwrite projected/output scratch for the next
        // state or window until both lanes complete this state's final read.
        CrossCoreSetFlag<4, PIPE_MTE3>(kStateFree);
    }

    static constexpr uint16_t kStateFree = 12;

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf_;
};
} // namespace KcpState
