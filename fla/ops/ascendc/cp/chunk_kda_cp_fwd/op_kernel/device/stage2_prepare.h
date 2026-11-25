#pragma once

#include "kernel_operator.h"

namespace KcpPrepare {
constexpr uint32_t kChunk = 64;
constexpr uint32_t kDim = 128;
constexpr uint32_t kHalf = kDim / 2;
constexpr uint32_t kBetaStride = 8;
constexpr float kLn2 = 0.6931471805599453f;
constexpr uint32_t kFeatureElements = kChunk * kDim;
constexpr uint32_t kScoreElements = kChunk * kChunk;
constexpr uint32_t kOutputElements = 4 * kFeatureElements + kScoreElements;

// All input/output bases refer to a single padded chunk for one head.
struct Window {
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR kk;
    GM_ADDR q;
    GM_ADDR qk;
    GM_ADDR factors;
    GM_ADDR decay;
};

class TriangularPrepare {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(solutions_, 2 * kChunk * kHalf * sizeof(float));
        pipe.InitBuffer(temp_, 7 * kHalf * sizeof(float));
        pipe.InitBuffer(bf_, kHalf * sizeof(bfloat16_t));
    }

    __aicore__ inline void Run(const Window& window, uint32_t rows, uint32_t lane, bool forward, float scale)
    {
        using namespace AscendC;
        GlobalTensor<bfloat16_t> k, v, q, out;
        GlobalTensor<float> gate, beta, kk, qk, decay;
        k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(window.k));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(window.v));
        gate.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(window.gate));
        beta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(window.beta));
        kk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(window.kk));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(window.factors));
        decay.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(window.decay));
        if (forward) {
            q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(window.q));
            qk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(window.qk));
        }
        const uint32_t col = lane * kHalf;
        auto w = solutions_.Get<float>();
        auto u = w[kChunk * kHalf];
        auto work = temp_.Get<float>();
        auto g = work[kHalf];
        auto last = work[2 * kHalf];
        auto factor = work[3 * kHalf];
        auto kRow = work[4 * kHalf];
        auto coeff = work[5 * kHalf];
        auto pad = work[6 * kHalf];
        auto bf = bf_.Get<bfloat16_t>();
        DataCopy(last, gate[(rows - 1) * kDim + col], kHalf);
        Duplicate(pad, 0.0f, kHalf);
        PipeBarrier<PIPE_ALL>();
        Exp2(factor, last);
        PipeBarrier<PIPE_ALL>();
        DataCopy(decay[col], factor, kHalf);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t offset = row * kDim + col;
            const uint32_t solveOffset = row * kHalf;
            auto wi = w[solveOffset];
            auto ui = u[solveOffset];
            // The previous window may have occupied the same GM beta slot.
            // Read the MTE3-produced row via MTE2 after the stage handoff.
            DataCopy(factor, beta[row * kBetaStride], kBetaStride);
            DataCopy(g, gate[offset], kHalf);
            DataCopy(bf, k[offset], kHalf);
            DataCopy(coeff, kk[row * kChunk], kChunk);
            PipeBarrier<PIPE_ALL>();
            const float b = factor.GetValue(0);
            Cast(kRow, bf, RoundMode::CAST_NONE, kHalf);
            Sub(work, last, g, kHalf);
            PipeBarrier<PIPE_V>();
            Exp2(factor, work);
            Mul(work, kRow, factor, kHalf);
            Write(out[2 * kFeatureElements + offset], work);

            Exp2(factor, g);
            Mul(wi, kRow, factor, kHalf);
            Muls(coeff, coeff, b, kChunk);
            PipeBarrier<PIPE_V>();
            Muls(wi, wi, b, kHalf);
            DataCopy(bf, v[offset], kHalf);
            PipeBarrier<PIPE_ALL>();
            Cast(ui, bf, RoundMode::CAST_NONE, kHalf);
            PipeBarrier<PIPE_V>();
            Muls(ui, ui, b, kHalf);
            // Each AIV solves its own columns. The FP32 rows stay live here
            // until every later row has read them. Round only the GM outputs.
            for (uint32_t j = 0; j < row; ++j) {
                PipeBarrier<PIPE_ALL>();
                const float f = -coeff.GetValue(j);
                Axpy(wi, w[j * kHalf], f, kHalf);
                Axpy(ui, u[j * kHalf], f, kHalf);
            }
            Write(out[offset], wi);
            Write(out[kFeatureElements + offset], ui);
            if (forward) {
                DataCopy(bf, q[offset], kHalf);
                PipeBarrier<PIPE_ALL>();
                Cast(work, bf, RoundMode::CAST_NONE, kHalf);
                PipeBarrier<PIPE_V>();
                Mul(work, work, factor, kHalf);
                PipeBarrier<PIPE_V>();
                Muls(work, work, scale, kHalf);
                Write(out[3 * kFeatureElements + offset], work);

                const uint32_t scoreOffset = row * kChunk + lane * (kChunk / 2);
                DataCopy(work, qk[scoreOffset], kChunk / 2);
                PipeBarrier<PIPE_ALL>();
                Muls(work, work, scale, kChunk / 2);
                Write(out[4 * kFeatureElements + scoreOffset], work, kChunk / 2);
            }
        }
        for (uint32_t row = rows; row < kChunk; ++row) {
            const uint32_t offset = row * kDim + col;
            Write(out[offset], pad);
            Write(out[kFeatureElements + offset], pad);
            Write(out[2 * kFeatureElements + offset], pad);
            if (forward) {
                Write(out[3 * kFeatureElements + offset], pad);
                const uint32_t scoreOffset = row * kChunk + lane * (kChunk / 2);
                Write(out[4 * kFeatureElements + scoreOffset], pad, kChunk / 2);
            }
        }
    }

private:
    __aicore__ inline void Exp2(
        const AscendC::LocalTensor<float>& out, const AscendC::LocalTensor<float>& in)
    {
        AscendC::Muls(out, in, kLn2, kHalf);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(out, out, kHalf);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Write(const AscendC::GlobalTensor<bfloat16_t>& out,
                                 const AscendC::LocalTensor<float>& in, uint32_t count = kHalf)
    {
        auto bf = bf_.Get<bfloat16_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(bf, in, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(out, bf, count);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> solutions_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf_;
};
} // namespace KcpPrepare
