#pragma once

#include "kernel_operator.h"

namespace KcpPreprocess {
constexpr uint32_t kDim = 128;
constexpr uint32_t kHalfDim = kDim / 2;
constexpr uint32_t kChunk = 64;
constexpr uint32_t kBetaStride = 8;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kEpsilon = 1.0e-6f;

struct RawWindow {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR g;
    GM_ADDR beta;
    GM_ADDR aLog;
    GM_ADDR bias;
    GM_ADDR qhat;
    GM_ADDR khat;
    GM_ADDR gate;
    GM_ADDR betaEff;
};

class Stage0 {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(fp_, 7 * kDim * sizeof(float));
        pipe.InitBuffer(bf_, kDim * sizeof(bfloat16_t));
    }

    __aicore__ inline void Bind(const RawWindow& view, bool forward)
    {
        if (forward) {
            q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(view.q));
            qhat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(view.qhat));
        }
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(view.k));
        khat_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(view.khat));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.beta));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.aLog));
        bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.bias));
        gate_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.gate));
        betaEff_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(view.betaEff));
    }

    // Bases express the first local token of the head. rowStride allows
    // native token-major BSND and the contiguous diagnostic layouts.
    __aicore__ inline void PrepareWindow(uint64_t qkBase, uint64_t vBase, uint64_t outBase,
        uint32_t qkStride, uint32_t vStride, uint32_t rows, uint32_t head, uint32_t lane, bool forward)
    {
        InitGate(head, lane * kHalfDim);
        const uint32_t mid = (rows + 1) / 2;
        auto running = fp_.Get<float>()[6 * kDim];
        AscendC::Duplicate(running, 0.0f, kHalfDim);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t col = lane * kHalfDim;
            Gate(running, (vBase + row * vStride) * kDim + col, (outBase + row) * kDim + col);
        }
        for (uint32_t row = lane == 0 ? 0 : mid; row < (lane == 0 ? mid : rows); ++row) {
            const uint64_t src = (qkBase + row * qkStride) * kDim;
            const uint64_t dst = (outBase + row) * kDim;
            if (forward) {
                Normalize(q_, qhat_, src, dst);
            }
            Normalize(k_, khat_, src, dst);
            Beta(vBase + row * vStride, outBase + row);
        }
    }

    __aicore__ inline void Run(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR bias,
                               GM_ADDR meta, GM_ADDR qhat, GM_ADDR khat, GM_ADDR gate, GM_ADDR betaEff)
    {
        auto* settings = reinterpret_cast<__gm__ const int32_t*>(meta);
        const uint32_t tokens = settings[0];
        const uint32_t heads = settings[1];
        const uint32_t seqs = settings[2];
        const uint32_t groups = settings[3];
        const uint32_t lane = AscendC::GetSubBlockIdx();
        const uint32_t owner = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        AscendC::TPipe pipe;
        Init(pipe);
        Bind({q, k, g, beta, aLog, bias, qhat, khat, gate, betaEff}, true);

        for (uint32_t task = owner; task < seqs * heads; task += groups) {
            const uint32_t head = task % heads;
            const uint32_t seq = task / heads;
            const uint32_t start = settings[4 + seq];
            const uint32_t end = settings[5 + seq];
            for (uint32_t chunk = start; chunk < end;) {
                const uint32_t remaining = end - chunk;
                const uint32_t size = remaining < kChunk ? remaining : kChunk;
                const uint64_t base = static_cast<uint64_t>(head) * tokens + chunk;
                PrepareWindow(base, base, base, 1, 1, size, head, lane, true);
                chunk += size;
            }
        }
    }

private:
    __aicore__ inline void Normalize(const AscendC::GlobalTensor<bfloat16_t>& input,
                                     const AscendC::GlobalTensor<bfloat16_t>& output, uint64_t src, uint64_t dst)
    {
        auto bf = bf_.Get<bfloat16_t>();
        auto fp = fp_.Get<float>();
        auto squares = fp[kDim];
        auto temp = fp[2 * kDim];
        auto norm = fp[3 * kDim];
        AscendC::DataCopy(bf, input[src], kDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(fp, bf, AscendC::RoundMode::CAST_NONE, kDim);
        AscendC::Duplicate(norm, 0.0f, kBetaStride);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(squares, fp, fp, kDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(norm, squares, temp, static_cast<int32_t>(kDim));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(norm, norm, kEpsilon, kBetaStride);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Rsqrt(norm, norm, kBetaStride);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(fp, fp, norm.GetValue(0), kDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(bf, fp, AscendC::RoundMode::CAST_RINT, kDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(output[dst], bf, kDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitGate(uint32_t head, uint32_t col)
    {
        auto fp = fp_.Get<float>();
        auto alpha = fp[4 * kDim];
        auto bias = fp[5 * kDim];
        AscendC::DataCopy(bias, bias_[head * kDim + col], kHalfDim);
        AscendC::Duplicate(alpha, aLog_.GetValue(head), kHalfDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(alpha, alpha, kHalfDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Sigmoid(const AscendC::LocalTensor<float>& row, uint32_t size)
    {
        AscendC::Muls(row, row, -1.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(row, row, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(row, row, 1.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(row, row, size);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Gate(const AscendC::LocalTensor<float>& running, uint64_t src, uint64_t dst)
    {
        auto fp = fp_.Get<float>();
        auto alpha = fp[4 * kDim];
        auto bias = fp[5 * kDim];
        AscendC::DataCopy(fp, g_[src], kHalfDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(fp, fp, bias, kHalfDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(fp, fp, alpha, kHalfDim);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid(fp, kHalfDim);
        AscendC::Muls(fp, fp, -5.0f * kLog2e, kHalfDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(running, running, fp, kHalfDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gate_[dst], running, kHalfDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Beta(uint64_t src, uint64_t dst)
    {
        auto fp = fp_.Get<float>();
        AscendC::Duplicate(fp, beta_.GetValue(src), kBetaStride);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid(fp, kBetaStride);
        AscendC::PipeBarrier<PIPE_ALL>();
        // The check output pads each scalar to its own DMA block.
        AscendC::DataCopy(betaEff_[dst * kBetaStride], fp, kBetaStride);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> fp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf_;
    AscendC::GlobalTensor<bfloat16_t> q_, k_, qhat_, khat_;
    AscendC::GlobalTensor<float> g_, beta_, aLog_, bias_, gate_, betaEff_;
};
} // namespace KcpPreprocess
