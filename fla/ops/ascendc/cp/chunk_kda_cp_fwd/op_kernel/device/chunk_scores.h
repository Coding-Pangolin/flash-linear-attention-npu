#pragma once

#include "chunk_gemm.h"
#include "stage2_prepare.h"

namespace KcpScore {
constexpr uint16_t kTile = 32;
constexpr uint16_t kHead = KcpPrepare::kDim;
constexpr uint32_t kChunk = KcpPrepare::kChunk;
constexpr uint32_t kInputTile = kTile * kHead;
constexpr uint32_t kScoreTile = kTile * kTile;
constexpr uint16_t kSecondLane = 16;
constexpr uint16_t kInputReady = 0;
constexpr uint16_t kInputGo = 1;
constexpr uint16_t kTileReady = 2;
constexpr uint16_t kScoreReady = 3;
constexpr uint16_t kScoreFree = 4;
constexpr uint16_t kDenseReady = 5;
constexpr uint16_t kDenseGo = 6;

struct Rect {
    int32_t row;
    int32_t rows;
    int32_t col;
    int32_t cols;
    int32_t anchor;
};

// For descending dyadic spans, keys are entirely before the query rows.
// Visit diagonal only for the inclusive QK score.
struct Rectangles {
    int32_t rows;
    int32_t half = kTile;
    int32_t start = 0;
    int32_t diagonal = 0;
    bool inclusive;

    __aicore__ inline bool Next(Rect& rect)
    {
        while (half != 0) {
            if (start + half < rows) {
                const int32_t row = start + half;
                const int32_t size = rows - row < half ? rows - row : half;
                rect = {row, size, start, half, row - 1};
                start += 2 * half;
                return true;
            }
            half /= 2;
            start = 0;
        }
        if (inclusive && diagonal < rows) {
            rect = {diagonal, 1, diagonal, 1, diagonal};
            ++diagonal;
            return true;
        }
        return false;
    }
};

class Factors {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(fp_, 4 * kHead * sizeof(float));
        pipe.InitBuffer(bf_, (kHead + kInputTile) * sizeof(bfloat16_t));
    }

    __aicore__ inline void Make(GM_ADDR operand, GM_ADDR logGate, const Rect& rect, bool right)
    {
        using namespace AscendC;
        GlobalTensor<bfloat16_t> input;
        GlobalTensor<float> gateGm;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(operand));
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logGate));
        auto f = fp_.Get<float>();
        auto values = f[kHead];
        auto gate = f[2 * kHead];
        auto anchor = f[3 * kHead];
        auto row = bf_.Get<bfloat16_t>();
        auto packed = row[kHead];
        auto bits = packed.ReinterpretCast<uint16_t>();
        auto rowBits = row.ReinterpretCast<uint16_t>();
        const int32_t start = right ? rect.col : rect.row;
        const int32_t n = right ? rect.cols : rect.rows;
        DataCopy(anchor, gateGm[rect.anchor * kHead], kHead);
        Duplicate(bits, uint16_t{0}, kInputTile);
        PipeBarrier<PIPE_ALL>();
        for (int32_t r = 0; r < n; ++r) {
            const uint32_t offset = (start + r) * kHead;
            DataCopy(row, input[offset], kHead);
            DataCopy(gate, gateGm[offset], kHead);
            PipeBarrier<PIPE_ALL>();
            if (right) {
                Sub(f, anchor, gate, kHead);
            } else {
                Sub(f, gate, anchor, kHead);
            }
            Cast(values, row, RoundMode::CAST_NONE, kHead);
            PipeBarrier<PIPE_V>();
            Muls(f, f, KcpPrepare::kLn2, kHead);
            PipeBarrier<PIPE_V>();
            Exp(f, f, kHead);
            PipeBarrier<PIPE_V>();
            Mul(values, values, f, kHead);
            PipeBarrier<PIPE_V>();
            if (right) {
                Cast(row, values, RoundMode::CAST_RINT, kHead);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t d = 0; d < kHead; ++d) {
                    bits.SetValue(d * kTile + r, rowBits.GetValue(d));
                }
            } else {
                Cast(packed[r * kHead], values, RoundMode::CAST_RINT, kHead);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void Write(GM_ADDR destination)
    {
        AscendC::GlobalTensor<bfloat16_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(destination));
        AscendC::DataCopy(out, bf_.Get<bfloat16_t>()[kHead], kInputTile);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf_;
};

struct Scratch {
    GM_ADDR left;
    GM_ADDR right;
    GM_ADDR tile;
};
constexpr uint32_t kScratchBytes = 2 * kInputTile * sizeof(bfloat16_t) + kScoreTile * sizeof(float);

__aicore__ inline Scratch MakeScratch(GM_ADDR base)
{
    return {base, base + kInputTile * sizeof(bfloat16_t), base + 2 * kInputTile * sizeof(bfloat16_t)};
}

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

__aicore__ inline void RunCube(KcpState::ChunkGemm& gemm, const Scratch& scratch, uint32_t rows, bool forward)
{
    WaitBoth<kInputReady>();
    // The waits execute on MTE2. Drain them before FIX can notify the
    // two AIVs that the peer's half of the normalized input is readable.
    AscendC::PipeBarrier<PIPE_ALL>();
    PublishBoth<kInputGo>();
    for (uint32_t branch = 0; branch < (forward ? 2U : 1U); ++branch) {
        Rectangles rects{static_cast<int32_t>(rows), kTile, 0, 0, branch != 0};
        Rect rect{};
        while (rects.Next(rect)) {
            WaitBoth<kTileReady>();
            gemm.Run<kTile, kTile, kHead>(scratch.left, scratch.right, scratch.tile);
            PublishBoth<kScoreReady>();
            WaitBoth<kScoreFree>();
        }
    }
    WaitBoth<kDenseReady>();
    AscendC::PipeBarrier<PIPE_ALL>();
    PublishBoth<kDenseGo>();
}

class ScoreVector {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        make_.Init(pipe);
        pipe.InitBuffer(fp_, (kScoreTile + kChunk * (kChunk / 2)) * sizeof(float));
    }

    __aicore__ inline void Run(const Scratch& scratch, const KcpPrepare::Window& prep, uint32_t rows,
                              uint32_t lane, bool forward)
    {
        using namespace AscendC;
        constexpr uint32_t kOwnedRows = kChunk / 2;
        const uint32_t rowBegin = lane * kOwnedRows;
        auto tile = fp_.Get<float>();
        auto dense = tile[kScoreTile];
        GlobalTensor<float> in;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scratch.tile));

        // The gate has two K-half writers. Synchronize through the owning
        // AIC before either lane reads full rows of normalized inputs/gate.
        CrossCoreSetFlag<4, PIPE_MTE3>(kInputReady);
        CrossCoreWaitFlag<4, PIPE_MTE2>(kInputGo);
        for (uint32_t branch = 0; branch < (forward ? 2U : 1U); ++branch) {
            Duplicate(dense, 0.0f, kOwnedRows * kChunk);
            Rectangles rects{static_cast<int32_t>(rows), kTile, 0, 0, branch != 0};
            Rect rect{};
            while (rects.Next(rect)) {
                const bool right = lane != 0;
                make_.Make(right || branch == 0 ? prep.k : prep.q, prep.gate, rect, right);
                make_.Write(right ? scratch.right : scratch.left);
                CrossCoreSetFlag<4, PIPE_MTE3>(kTileReady);
                CrossCoreWaitFlag<4, PIPE_MTE2>(kScoreReady);
                if (rect.row + rect.rows > static_cast<int32_t>(rowBegin) &&
                    rect.row < static_cast<int32_t>(rowBegin + kOwnedRows)) {
                    DataCopy(tile, in, kScoreTile);
                    PipeBarrier<PIPE_ALL>();
                    for (int32_t r = 0; r < rect.rows; ++r) {
                        const int32_t local = rect.row + r - rowBegin;
                        if (local >= 0 && local < static_cast<int32_t>(kOwnedRows)) {
                            for (int32_t c = 0; c < rect.cols; ++c) {
                                dense.SetValue(local * kChunk + rect.col + c, tile.GetValue(r * kTile + c));
                            }
                        }
                    }
                }
                CrossCoreSetFlag<4, PIPE_MTE2>(kScoreFree);
                PipeBarrier<PIPE_ALL>();
            }
            GlobalTensor<float> dest;
            dest.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(branch == 0 ? prep.kk : prep.qk));
            PipeBarrier<PIPE_ALL>();
            DataCopy(dest[rowBegin * kChunk], dense, kOwnedRows * kChunk);
            PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<4, PIPE_MTE3>(kDenseReady);
        CrossCoreWaitFlag<4, PIPE_MTE2>(kDenseGo);
    }

private:
    Factors make_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp_;
};
} // namespace KcpScore
