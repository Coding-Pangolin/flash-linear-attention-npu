/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — BNSD + GVA.
 * tiling key 0: AIV scalar fallback
 * tiling key 1: MIX_AIC_1_2 Cube (Catlass BlockMmad ×2, kneg shared GM tile)
 *
 * Aligns with GPU Triton chunk_kda_fwd_kernel_intra_sub_chunk.
 */

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;
using _16 = tla::Int<16>;
using _128 = tla::Int<128>;

namespace {
constexpr float LN2 = 0.6931471805599453f;
// Align Cube prep clamp with chunk_kda_fwd (exp2 domain ≈ ±80).
constexpr float EXP2_CLAMP = 80.0f;
constexpr float EXP_INPUT_MAX = EXP2_CLAMP * LN2;
constexpr float EXP_INPUT_MIN = -EXP2_CLAMP * LN2;
constexpr float FP16_MAX = 65504.0f;
constexpr uint32_t MAX_K = 256;
constexpr uint32_t MAX_BC = 16;
constexpr uint32_t SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t SCORE_PLANES = 3;
constexpr uint32_t C_PLANES = 2;
constexpr uint32_t PLANE_QG = 0;
constexpr uint32_t PLANE_W = 1;
constexpr uint32_t PLANE_KG = 2;
constexpr uint32_t PLANE_AQK = 0;
constexpr uint32_t PLANE_AKK = 1;
constexpr uint32_t SOLVE_X = 0;
constexpr uint32_t SOLVE_Y0 = 1;
constexpr uint32_t SOLVE_TMP = 2;
constexpr uint32_t SOLVE_Y1 = 3;
constexpr uint32_t SOLVE_PLANES = 4;
constexpr uint32_t MCH_ITERS = 3;
constexpr uint8_t FLAG_DONE = 2;
constexpr uint8_t FLAG_READY = 4;
constexpr uint8_t FLAG_SOLVE_DONE = 6;
constexpr uint8_t FLAG_SOLVE_READY = 8;
constexpr uint32_t EVT_MTE2_V = 0;
constexpr uint32_t EVT_V_S = 4;
constexpr uint32_t EVT_S_V = 5;
constexpr uint32_t EVT_V_MTE3 = 6;
constexpr uint32_t EVT_MTE3_V = 7;
constexpr uint32_t EVT_MTE3_MTE2 = 3;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
using KdaSolveDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
static_assert(!KdaSolveDispatchPolicy::USE_HF32_MODE, "intra-sub-chunk MCH must use IEEE FP32 Cube");
using KdaL1TileShape = tla::Shape<_16, _16, _128>;
using KdaL0TileShape = KdaL1TileShape;
using KdaSolveL1TileShape = tla::Shape<_16, _16, _16>;
using KdaSolveL0TileShape = KdaSolveL1TileShape;

// ---------------------------------------------------------------------------
// Scalar fallback (tiling key 0)
// ---------------------------------------------------------------------------
template <typename T>
class ChunkKdaFwdIntraSubChunkScalarKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR aqk, GM_ADDR akkd, const ChunkKdaFwdIntraSubChunkTilingData &tiling, TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(beta));
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aqk));
        akkd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(akkd));
        pipe_ = pipe;

        batch_ = static_cast<uint64_t>(tiling.batch);
        t_ = static_cast<uint64_t>(tiling.t);
        h_ = static_cast<uint64_t>(tiling.h);
        hv_ = static_cast<uint64_t>(tiling.hv);
        kDim_ = static_cast<uint64_t>(tiling.k);
        bt_ = static_cast<uint64_t>(tiling.chunkSize);
        bc_ = static_cast<uint64_t>(tiling.subChunkSize);
        nc_ = static_cast<uint64_t>(tiling.numSubChunks);
        totalTasks_ = static_cast<uint64_t>(tiling.totalTasks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        hasVarlen_ = tiling.hasCuSeqlens != 0;
        scale_ = tiling.scale;
        group_ = (h_ == 0) ? 1 : (hv_ / h_);

        pipe_->InitBuffer(qBuf_, MAX_BC * MAX_K * sizeof(float));
        pipe_->InitBuffer(kBuf_, MAX_BC * MAX_K * sizeof(float));
        pipe_->InitBuffer(gBuf_, MAX_BC * MAX_K * sizeof(float));
        pipe_->InitBuffer(epBuf_, MAX_BC * MAX_K * sizeof(float));
        pipe_->InitBuffer(enBuf_, MAX_BC * MAX_K * sizeof(float));
        pipe_->InitBuffer(midBuf_, MAX_K * sizeof(float));
        pipe_->InitBuffer(betaBuf_, MAX_BC * sizeof(float));
        pipe_->InitBuffer(aqkBuf_, MAX_BC * MAX_BC * sizeof(float));
        pipe_->InitBuffer(akkBuf_, MAX_BC * MAX_BC * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, MAX_BC * sizeof(float));
        pipe_->InitBuffer(inBuf_, MAX_K * sizeof(T));
        pipe_->InitBuffer(i64Buf_, 32);
        pipe_->InitBuffer(scalarBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        if (usedCoreNum_ == 0 || totalTasks_ == 0 || kDim_ == 0 || bc_ == 0 || bc_ > MAX_BC || kDim_ > MAX_K ||
            h_ == 0 || hv_ == 0 || hv_ < h_ || (hv_ % h_) != 0) {
            return;
        }
        for (uint64_t task = coreIdx; task < totalTasks_; task += usedCoreNum_) {
            ProcessChunk(task);
        }
    }

private:
    __aicore__ inline void SyncVS()
    {
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVT_V_S);
        WaitFlag<HardEvent::V_S>(EVT_V_S);
    }
    __aicore__ inline void SyncSV()
    {
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint64_t QkRowOff(uint64_t b, uint64_t h, uint64_t t) const
    {
        return ((b * h_ + h) * t_ + t) * kDim_;
    }
    __aicore__ inline uint64_t HvRowOff(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * hv_ + hv) * t_ + t) * kDim_;
    }
    __aicore__ inline uint64_t BetaOff(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * hv_ + hv) * t_ + t;
    }
    __aicore__ inline uint64_t AqkOff(uint64_t b, uint64_t hv, uint64_t t, uint64_t col) const
    {
        return ((b * hv_ + hv) * t_ + t) * bt_ + col;
    }
    __aicore__ inline uint64_t AkkdOff(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return ((b * hv_ + hv) * t_ + t) * bc_;
    }
    __aicore__ inline void RunExp2(LocalTensor<float> tensor, uint32_t count)
    {
        SyncSV();
        Mins(tensor, tensor, EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
        SyncVS();
    }
    __aicore__ inline int64_t LoadI64(GlobalTensor<int64_t> &src, uint64_t off)
    {
        LocalTensor<int64_t> s = i64Buf_.Get<int64_t>();
        DataCopyParams p{1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0};
        DataCopyPadParams pad{false, 0, 0, 0};
        DataCopyPad(s, src[off], p, pad);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        SyncVS();
        return s.GetValue(0);
    }
    __aicore__ inline float LoadScalarT(GlobalTensor<T> &src, uint64_t off)
    {
        LocalTensor<float> dst = scalarBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyParams p{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(dst, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Adds(dst, dst, 0.0f, 1);
        } else {
            LocalTensor<T> in = inBuf_.Get<T>();
            DataCopyParams p{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(in, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Cast(dst, in, RoundMode::CAST_NONE, 1);
        }
        SyncVS();
        return dst.GetValue(0);
    }
    __aicore__ inline void LoadRow(LocalTensor<float> base, uint32_t rowOffset, GlobalTensor<T> &src, uint64_t off,
                                   uint64_t n)
    {
        LocalTensor<float> row = midBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyParams p{1, static_cast<uint16_t>(n * sizeof(float)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(row, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Adds(row, row, 0.0f, static_cast<uint32_t>(n));
        } else {
            LocalTensor<T> in = inBuf_.Get<T>();
            DataCopyParams p{1, static_cast<uint16_t>(n * sizeof(T)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(in, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Cast(row, in, RoundMode::CAST_NONE, static_cast<uint32_t>(n));
        }
        PipeBarrier<PIPE_V>();
        SyncVS();
        for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
            base.SetValue(rowOffset + i, row.GetValue(i));
        }
    }
    __aicore__ inline void LoadMidRow(GlobalTensor<T> &src, uint64_t off, uint64_t n)
    {
        LocalTensor<float> mid = midBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyParams p{1, static_cast<uint16_t>(n * sizeof(float)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(mid, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Adds(mid, mid, 0.0f, static_cast<uint32_t>(n));
        } else {
            LocalTensor<T> in = inBuf_.Get<T>();
            DataCopyParams p{1, static_cast<uint16_t>(n * sizeof(T)), 0, 0};
            DataCopyPadParams pad{false, 0, 0, 0};
            DataCopyPad(in, src[off], p, pad);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Cast(mid, in, RoundMode::CAST_NONE, static_cast<uint32_t>(n));
        }
        PipeBarrier<PIPE_V>();
        SyncVS();
    }
    __aicore__ inline void StoreAqkRow(uint64_t b, uint64_t hv, uint64_t t, uint64_t col, LocalTensor<float> row)
    {
        const uint64_t base = AqkOff(b, hv, t, col);
        LocalTensor<float> sf = scalarBuf_.Get<float>();
        LocalTensor<T> st = inBuf_.Get<T>();
        for (uint32_t j = 0; j < static_cast<uint32_t>(bc_); ++j) {
            sf.SetValue(0, row.GetValue(j));
            SyncSV();
            if constexpr (IsSameType<T, float>::value) {
                aqk_.SetValue(base + j, sf.GetValue(0));
            } else {
                Cast(st, sf, RoundMode::CAST_RINT, 1);
                SyncVS();
                aqk_.SetValue(base + j, st.GetValue(0));
            }
        }
    }
    __aicore__ inline void StoreAkkdRow(uint64_t b, uint64_t hv, uint64_t t, LocalTensor<float> row)
    {
        const uint64_t base = AkkdOff(b, hv, t);
        for (uint32_t j = 0; j < static_cast<uint32_t>(bc_); ++j) {
            akkd_.SetValue(base + j, row.GetValue(j));
        }
    }
    __aicore__ inline void ClearMatrix(LocalTensor<float> mat, uint32_t elems)
    {
        for (uint32_t i = 0; i < elems; ++i) {
            mat.SetValue(i, 0.0f);
        }
    }

    __aicore__ inline void ResolveChunk(uint64_t iChunk, uint64_t &bos, uint64_t &localT, uint64_t &localChunk,
                                        uint64_t &bIdx)
    {
        bos = 0;
        localT = t_;
        localChunk = iChunk;
        bIdx = 0;
        if (hasVarlen_) {
            const int64_t seqId = LoadI64(chunkIndices_, iChunk * 2);
            localChunk = static_cast<uint64_t>(LoadI64(chunkIndices_, iChunk * 2 + 1));
            bos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId)));
            const uint64_t eos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId) + 1));
            localT = eos - bos;
            bIdx = 0;
        }
    }

    __aicore__ inline void ProcessSub(uint64_t bIdx, uint64_t iH, uint64_t iHv, uint64_t bos, uint64_t localT,
                                      uint64_t localChunk, uint64_t iSub)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        if (iTi >= localT) {
            return;
        }
        const uint64_t valid = (iTi + bc_ <= localT) ? bc_ : (localT - iTi);
        const uint64_t midRel = (bc_ / 2 < localT - iTi) ? (bc_ / 2) : (localT - iTi - 1);

        LocalTensor<float> q = qBuf_.Get<float>();
        LocalTensor<float> k = kBuf_.Get<float>();
        LocalTensor<float> g = gBuf_.Get<float>();
        LocalTensor<float> ep = epBuf_.Get<float>();
        LocalTensor<float> en = enBuf_.Get<float>();
        LocalTensor<float> mid = midBuf_.Get<float>();
        LocalTensor<float> beta = betaBuf_.Get<float>();
        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();

        ClearMatrix(q, static_cast<uint32_t>(bc_ * kDim_));
        ClearMatrix(k, static_cast<uint32_t>(bc_ * kDim_));
        ClearMatrix(g, static_cast<uint32_t>(bc_ * kDim_));
        ClearMatrix(ep, static_cast<uint32_t>(bc_ * kDim_));
        ClearMatrix(en, static_cast<uint32_t>(bc_ * kDim_));
        ClearMatrix(beta, static_cast<uint32_t>(bc_));
        ClearMatrix(aqk, static_cast<uint32_t>(bc_ * bc_));
        ClearMatrix(akk, static_cast<uint32_t>(bc_ * bc_));

        for (uint64_t r = 0; r < valid; ++r) {
            const uint64_t tok = bos + iTi + r;
            const uint32_t rowOff = static_cast<uint32_t>(r * kDim_);
            LoadRow(q, rowOff, q_, QkRowOff(bIdx, iH, tok), kDim_);
            LoadRow(k, rowOff, k_, QkRowOff(bIdx, iH, tok), kDim_);
            LoadRow(g, rowOff, g_, HvRowOff(bIdx, iHv, tok), kDim_);
            beta.SetValue(static_cast<uint32_t>(r), LoadScalarT(beta_, BetaOff(bIdx, iHv, tok)));
        }

        LoadMidRow(g_, HvRowOff(bIdx, iHv, bos + iTi + midRel), kDim_);
        mid = midBuf_.Get<float>();
        for (uint64_t r = 0; r < valid; ++r) {
            const uint32_t rowOff = static_cast<uint32_t>(r * kDim_);
            for (uint32_t d = 0; d < static_cast<uint32_t>(kDim_); ++d) {
                const float gm = g.GetValue(rowOff + d) - mid.GetValue(d);
                ep.SetValue(rowOff + d, gm * LN2);
                en.SetValue(rowOff + d, (-gm) * LN2);
            }
        }
        SyncSV();
        RunExp2(ep, static_cast<uint32_t>(valid * kDim_));
        RunExp2(en, static_cast<uint32_t>(valid * kDim_));

        for (uint64_t i = 0; i < valid; ++i) {
            for (uint64_t j = 0; j < valid; ++j) {
                float sa = 0.0f;
                float sk = 0.0f;
                for (uint64_t d = 0; d < kDim_; ++d) {
                    const uint32_t ii = static_cast<uint32_t>(i * kDim_ + d);
                    const uint32_t jj = static_cast<uint32_t>(j * kDim_ + d);
                    const float gqi = ep.GetValue(ii);
                    const float gkj = en.GetValue(jj);
                    sa += q.GetValue(ii) * gqi * k.GetValue(jj) * gkj;
                    sk += k.GetValue(ii) * gqi * k.GetValue(jj) * gkj;
                }
                sa *= scale_;
                sk *= beta.GetValue(static_cast<uint32_t>(i));
                if (i >= j) {
                    aqk.SetValue(static_cast<uint32_t>(i * bc_ + j), sa);
                }
                if (i > j) {
                    akk.SetValue(static_cast<uint32_t>(i * bc_ + j), sk);
                }
            }
        }
        for (uint64_t i = 0; i < valid; ++i) {
            for (uint64_t j = 0; j < i; ++j) {
                const uint32_t idx = static_cast<uint32_t>(i * bc_ + j);
                akk.SetValue(idx, -akk.GetValue(idx));
            }
        }
        for (uint64_t i = 2; i < valid; ++i) {
            for (uint64_t j = 0; j < bc_; ++j) {
                float v = (j < i) ? akk.GetValue(static_cast<uint32_t>(i * bc_ + j)) : 0.0f;
                tmp.SetValue(static_cast<uint32_t>(j), v);
            }
            for (uint64_t j = 0; j < i; ++j) {
                float acc = tmp.GetValue(static_cast<uint32_t>(j));
                for (uint64_t p = 0; p < i; ++p) {
                    acc += tmp.GetValue(static_cast<uint32_t>(p)) *
                           akk.GetValue(static_cast<uint32_t>(p * bc_ + j));
                }
                tmp.SetValue(static_cast<uint32_t>(j), acc);
            }
            for (uint64_t j = 0; j < bc_; ++j) {
                float v = (j < i) ? tmp.GetValue(static_cast<uint32_t>(j)) : 0.0f;
                akk.SetValue(static_cast<uint32_t>(i * bc_ + j), v);
            }
        }
        for (uint64_t i = 0; i < valid; ++i) {
            const uint32_t diag = static_cast<uint32_t>(i * bc_ + i);
            akk.SetValue(diag, akk.GetValue(diag) + 1.0f);
        }
        for (uint64_t r = 0; r < valid; ++r) {
            const uint64_t tok = bos + iTi + r;
            for (uint64_t j = 0; j < bc_; ++j) {
                tmp.SetValue(static_cast<uint32_t>(j), aqk.GetValue(static_cast<uint32_t>(r * bc_ + j)));
            }
            StoreAqkRow(bIdx, iHv, tok, iSub * bc_, tmp);
            for (uint64_t j = 0; j < bc_; ++j) {
                tmp.SetValue(static_cast<uint32_t>(j), akk.GetValue(static_cast<uint32_t>(r * bc_ + j)));
            }
            StoreAkkdRow(bIdx, iHv, tok, tmp);
        }
    }

    __aicore__ inline void ProcessChunk(uint64_t task)
    {
        const uint64_t bhv = batch_ * hv_;
        const uint64_t iBhv = task % bhv;
        const uint64_t iChunk = task / bhv;
        const uint64_t iB = iBhv / hv_;
        const uint64_t iHv = iBhv % hv_;
        const uint64_t iH = iHv / group_;
        uint64_t bos = 0, localT = t_, localChunk = iChunk, bIdx = iB;
        ResolveChunk(iChunk, bos, localT, localChunk, bIdx);
        if (!hasVarlen_) {
            bIdx = iB;
        }
        for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
            ProcessSub(bIdx, iH, iHv, bos, localT, localChunk, iSub);
        }
    }

    GlobalTensor<T> q_, k_, g_, beta_, aqk_;
    GlobalTensor<float> akkd_;
    GlobalTensor<int64_t> cuSeqlens_, chunkIndices_;
    TPipe *pipe_ = nullptr;
    TBuf<> qBuf_, kBuf_, gBuf_, epBuf_, enBuf_, midBuf_, betaBuf_, aqkBuf_, akkBuf_, tmpBuf_, inBuf_, i64Buf_,
        scalarBuf_;
    uint64_t batch_ = 0, t_ = 0, h_ = 0, hv_ = 0, kDim_ = 0, bt_ = 0, bc_ = 0, nc_ = 0, totalTasks_ = 0,
             usedCoreNum_ = 0, group_ = 1;
    bool hasVarlen_ = false;
    float scale_ = 1.0f;
};

// ---------------------------------------------------------------------------
// Cube path (tiling key 1)
// ---------------------------------------------------------------------------
template <typename T>
class ChunkKdaFwdIntraSubChunkCubeKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR aqk, GM_ADDR akkd, GM_ADDR userWS,
                                const ChunkKdaFwdIntraSubChunkTilingData &tiling, TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(beta));
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aqk));
        akkd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(akkd));
        scoreWs_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(userWS));
        cmatWs_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWS + tiling.scoreScratchBytes));
        {
            const uint64_t cBytes = static_cast<uint64_t>(tiling.usedCoreNum) *
                                   static_cast<uint64_t>(tiling.scoreQueueDepth == 0 ? SCORE_QUEUE_DEPTH
                                                                                    : tiling.scoreQueueDepth) *
                                   C_PLANES * MAX_BC * MAX_BC * sizeof(float);
            solveWs_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(userWS + tiling.scoreScratchBytes + static_cast<int64_t>(cBytes)));
        }
        pipe_ = pipe;

        batch_ = static_cast<uint64_t>(tiling.batch);
        t_ = static_cast<uint64_t>(tiling.t);
        h_ = static_cast<uint64_t>(tiling.h);
        hv_ = static_cast<uint64_t>(tiling.hv);
        kDim_ = static_cast<uint64_t>(tiling.k);
        bt_ = static_cast<uint64_t>(tiling.chunkSize);
        bc_ = static_cast<uint64_t>(tiling.subChunkSize);
        nc_ = static_cast<uint64_t>(tiling.numSubChunks);
        totalTasks_ = static_cast<uint64_t>(tiling.totalTasks);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        depth_ = static_cast<uint64_t>(tiling.scoreQueueDepth);
        if (depth_ == 0) {
            depth_ = SCORE_QUEUE_DEPTH;
        }
        hasVarlen_ = tiling.hasCuSeqlens != 0;
        scale_ = tiling.scale;
        group_ = (h_ == 0) ? 1 : (hv_ / h_);

        if ASCEND_IS_AIV {
            const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            coreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            coreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }

        if (pipe_ != nullptr) {
            if ASCEND_IS_AIV {
                pipe_->InitBuffer(vecBuf_, MAX_BC * MAX_K * 6 * sizeof(float));
                pipe_->InitBuffer(midBuf_, MAX_K * sizeof(float));
                pipe_->InitBuffer(betaBuf_, MAX_BC * sizeof(float));
                pipe_->InitBuffer(aqkBuf_, MAX_BC * MAX_BC * sizeof(float));
                pipe_->InitBuffer(akkBuf_, MAX_BC * MAX_BC * sizeof(float));
                pipe_->InitBuffer(tmpBuf_, MAX_BC * sizeof(float));
                pipe_->InitBuffer(inBuf_, MAX_K * sizeof(T));
                pipe_->InitBuffer(i64Buf_, 32);
                pipe_->InitBuffer(scalarBuf_, 32);
                pipe_->InitBuffer(zeroBuf_, MAX_BC * MAX_K * sizeof(T));
            }
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        if (usedCoreNum_ == 0 || totalTasks_ == 0 || kDim_ == 0 || bc_ != MAX_BC || kDim_ > MAX_K || h_ == 0 ||
            hv_ == 0 || hv_ < h_ || (hv_ % h_) != 0) {
            return;
        }
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            ProcessChunkAiv(task);
        }
    }

    __aicore__ inline void ProcessAic()
    {
        if (usedCoreNum_ == 0 || totalTasks_ == 0 || kDim_ == 0 || bc_ != MAX_BC || kDim_ > MAX_K) {
            return;
        }
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            ProcessChunkAic(task);
        }
    }

private:
    Catlass::Arch::CrossCoreFlag readyFlag_{FLAG_READY};
    Catlass::Arch::CrossCoreFlag doneFlag_{FLAG_DONE};
    Catlass::Arch::CrossCoreFlag solveReadyFlag_{FLAG_SOLVE_READY};
    Catlass::Arch::CrossCoreFlag solveDoneFlag_{FLAG_SOLVE_DONE};

    __aicore__ inline void SyncVS()
    {
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVT_V_S);
        WaitFlag<HardEvent::V_S>(EVT_V_S);
    }
    __aicore__ inline void SyncSV()
    {
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t QkRowOff(uint64_t b, uint64_t h, uint64_t tok) const
    {
        return ((b * h_ + h) * t_ + tok) * kDim_;
    }
    __aicore__ inline uint64_t HvRowOff(uint64_t b, uint64_t hv, uint64_t tok) const
    {
        return ((b * hv_ + hv) * t_ + tok) * kDim_;
    }
    __aicore__ inline uint64_t BetaOff(uint64_t b, uint64_t hv, uint64_t tok) const
    {
        return (b * hv_ + hv) * t_ + tok;
    }
    __aicore__ inline uint64_t AqkOff(uint64_t b, uint64_t hv, uint64_t tok, uint64_t col) const
    {
        return ((b * hv_ + hv) * t_ + tok) * bt_ + col;
    }
    __aicore__ inline uint64_t AkkdOff(uint64_t b, uint64_t hv, uint64_t tok) const
    {
        return ((b * hv_ + hv) * t_ + tok) * bc_;
    }
    __aicore__ inline uint64_t ScoreOff(uint64_t slot, uint64_t plane, uint64_t row, uint64_t d = 0) const
    {
        return (((coreIdx_ * depth_ + slot) * SCORE_PLANES + plane) * bc_ + row) * kDim_ + d;
    }
    __aicore__ inline uint64_t CmatOff(uint64_t slot, uint64_t plane, uint64_t row = 0, uint64_t col = 0) const
    {
        return (((coreIdx_ * depth_ + slot) * C_PLANES + plane) * bc_ + row) * bc_ + col;
    }
    __aicore__ inline uint64_t SolveOff(uint64_t slot, uint64_t plane, uint64_t row = 0, uint64_t col = 0) const
    {
        return (((coreIdx_ * depth_ + slot) * SOLVE_PLANES + plane) * bc_ + row) * bc_ + col;
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

    __aicore__ inline int64_t LoadI64(GlobalTensor<int64_t> &src, uint64_t off)
    {
        LocalTensor<int64_t> s = i64Buf_.Get<int64_t>();
        DataCopyParams p{1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0};
        DataCopyPadParams pad{false, 0, 0, 0};
        DataCopyPad(s, src[off], p, pad);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        SyncVS();
        return s.GetValue(0);
    }

    __aicore__ inline float LoadScalarT(GlobalTensor<T> &src, uint64_t off)
    {
        LocalTensor<float> dst = scalarBuf_.Get<float>();
        LocalTensor<T> in = inBuf_.Get<T>();
        DataCopyParams p{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
        DataCopyPadParams pad{false, 0, 0, 0};
        DataCopyPad(in, src[off], p, pad);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(dst, in, RoundMode::CAST_NONE, 1);
        SyncVS();
        return dst.GetValue(0);
    }

    __aicore__ inline void LoadMidRow(uint64_t off)
    {
        LocalTensor<float> mid = midBuf_.Get<float>();
        LocalTensor<T> in = inBuf_.Get<T>();
        CopyVectorIn(in, g_, off, kDim_);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(mid, in, RoundMode::CAST_NONE, static_cast<uint32_t>(kDim_));
        PipeBarrier<PIPE_V>();
        SyncVS();
    }

    __aicore__ inline void ClampExp(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClampFp16(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<T, half>::value) {
            Mins(tensor, tensor, FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -FP16_MAX, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ZeroScorePlaneRows(uint64_t slot, uint64_t plane, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint64_t nRows = rowEnd - rowBegin;
        const uint32_t elems = static_cast<uint32_t>(nRows * kDim_);
        LocalTensor<T> z = zeroBuf_.Get<T>();
        Duplicate(z, static_cast<T>(0), elems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(scoreWs_, ScoreOff(slot, plane, rowBegin, 0), z, elems);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // Max tile rows fitting vecBuf: 6 float planes + 3 typed planes.
    __aicore__ inline uint64_t PrepMaxTileRows() const
    {
        const uint64_t bytesPerElem = 6ull * sizeof(float) + 3ull * sizeof(T);
        const uint64_t budget = static_cast<uint64_t>(MAX_BC) * MAX_K * 6ull * sizeof(float);
        uint64_t maxElems = budget / bytesPerElem;
        if (maxElems > static_cast<uint64_t>(MAX_BC) * kDim_) {
            maxElems = static_cast<uint64_t>(MAX_BC) * kDim_;
        }
        uint64_t rows = maxElems / kDim_;
        if (rows == 0) {
            rows = 1;
        }
        return rows;
    }

    __aicore__ inline void ResolveChunk(uint64_t iChunk, uint64_t iB, uint64_t &bos, uint64_t &localT,
                                        uint64_t &localChunk, uint64_t &bIdx)
    {
        bos = 0;
        localT = t_;
        localChunk = iChunk;
        bIdx = iB;
        if (hasVarlen_) {
            const int64_t seqId = LoadI64(chunkIndices_, iChunk * 2);
            localChunk = static_cast<uint64_t>(LoadI64(chunkIndices_, iChunk * 2 + 1));
            bos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId)));
            const uint64_t eos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId) + 1));
            localT = eos - bos;
            bIdx = 0;
        }
    }

    __aicore__ inline void DecodeTask(uint64_t task, uint64_t &iB, uint64_t &iHv, uint64_t &iH, uint64_t &iChunk)
    {
        const uint64_t bhv = batch_ * hv_;
        const uint64_t iBhv = task % bhv;
        iChunk = task / bhv;
        iB = iBhv / hv_;
        iHv = iBhv % hv_;
        iH = iHv / group_;
    }

    // P1: multi-row batched prep (align PrepareScoreFactorsBulk) — one HardEvent set per tile.
    __aicore__ inline void PrepareSub(uint64_t bIdx, uint64_t iH, uint64_t iHv, uint64_t bos, uint64_t localT,
                                      uint64_t localChunk, uint64_t iSub, uint64_t slot, uint64_t subBlockIdx,
                                      uint64_t subBlockNum)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        const bool empty = (iTi >= localT);
        const uint64_t valid = empty ? 0 : ((iTi + bc_ <= localT) ? bc_ : (localT - iTi));
        // Contiguous half-range split (cache-friendly); pad to bc_ so both AIVs cover full scratch rows.
        const uint64_t rowBegin = (bc_ * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (bc_ * (subBlockIdx + 1)) / subBlockNum;

        if (empty || valid == 0 || rowBegin >= rowEnd) {
            ZeroScorePlaneRows(slot, PLANE_QG, rowBegin, rowEnd);
            ZeroScorePlaneRows(slot, PLANE_W, rowBegin, rowEnd);
            ZeroScorePlaneRows(slot, PLANE_KG, rowBegin, rowEnd);
            return;
        }

        const uint64_t midRel = (bc_ / 2 < localT - iTi) ? (bc_ / 2) : (localT - iTi - 1);
        LoadMidRow(HvRowOff(bIdx, iHv, bos + iTi + midRel));
        LocalTensor<float> mid = midBuf_.Get<float>();

        const uint64_t maxTile = PrepMaxTileRows();
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxTile) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxTile) {
                tileRows = maxTile;
            }

            // Rows in [valid, bc_) stay zero — split tile into live + pad.
            uint64_t liveRows = 0;
            if (tileRow < valid) {
                liveRows = valid - tileRow;
                if (liveRows > tileRows) {
                    liveRows = tileRows;
                }
            }
            if (liveRows < tileRows) {
                ZeroScorePlaneRows(slot, PLANE_QG, tileRow + liveRows, tileRow + tileRows);
                ZeroScorePlaneRows(slot, PLANE_W, tileRow + liveRows, tileRow + tileRows);
                ZeroScorePlaneRows(slot, PLANE_KG, tileRow + liveRows, tileRow + tileRows);
            }
            if (liveRows == 0) {
                continue;
            }

            const uint64_t elems = liveRows * kDim_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp = arena;
            LocalTensor<float> kFp = arena[elems];
            LocalTensor<float> gFp = arena[2 * elems];
            LocalTensor<float> expP = arena[3 * elems];
            LocalTensor<float> expN = arena[4 * elems];
            LocalTensor<float> out = arena[5 * elems];
            const uint64_t typedOff = (6 * elems * sizeof(float) + sizeof(T) - 1) / sizeof(T);
            LocalTensor<T> typed = vecBuf_.Get<T>()[typedOff];
            LocalTensor<T> qT = typed;
            LocalTensor<T> kT = typed[elems];
            LocalTensor<T> outT = typed[2 * elems];

            const uint64_t tok0 = bos + iTi + tileRow;
            CopyVectorIn(qT, q_, QkRowOff(bIdx, iH, tok0), elems);
            CopyVectorIn(kT, k_, QkRowOff(bIdx, iH, tok0), elems);
            CopyVectorIn(outT, g_, HvRowOff(bIdx, iHv, tok0), elems);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Cast(qFp, qT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp, kT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(gFp, outT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            for (uint64_t row = 0; row < liveRows; ++row) {
                Sub(expP[row * kDim_], gFp[row * kDim_], mid, static_cast<uint32_t>(kDim_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expP, expP, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Muls(expN, expP, -1.0f, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampExp(expP, static_cast<uint32_t>(elems));
            ClampExp(expN, static_cast<uint32_t>(elems));
            Exp(expP, expP, static_cast<uint32_t>(elems));
            Exp(expN, expN, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            // QG
            Mul(out, qFp, expP, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outT, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_QG, tileRow, 0), outT, elems);
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);

            // W (kpos)
            Mul(out, kFp, expP, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outT, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_W, tileRow, 0), outT, elems);
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);

            // KG (kneg)
            Mul(out, kFp, expN, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outT, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_KG, tileRow, 0), outT, elems);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        }
    }

    // HARD: same blockKNeg for both MMADs (GM one KG plane). L1 residence = profile target for c6.
    // ElementA/B = T (bf16/fp16), ElementC = fp32 — same as chunk_kda_fwd::ComputeRawAqkAkkCubeBlock.
    __aicore__ inline void ComputeMmad(uint64_t slot)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, kDim_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);
        Catlass::GemmCoord shape{static_cast<uint32_t>(bc_), static_cast<uint32_t>(bc_),
                                 static_cast<uint32_t>(kDim_)};

        auto tensorQg = tla::MakeTensor(scoreWs_[ScoreOff(slot, PLANE_QG, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(scoreWs_[ScoreOff(slot, PLANE_W, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorKg = tla::MakeTensor(scoreWs_[ScoreOff(slot, PLANE_KG, 0, 0)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorAqk =
            tla::MakeTensor(cmatWs_[CmatOff(slot, PLANE_AQK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk =
            tla::MakeTensor(cmatWs_[CmatOff(slot, PLANE_AKK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});

        auto blockQg = GetTile(tensorQg, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));

        // MMAD1: qg @ knegᵀ → Aqk_raw  (loads B=kneg into L1/L0)
        blockMmad(blockQg, blockKg, blockAqk, shape);
        PipeBarrier<PIPE_ALL>();
        // MMAD2: kpos @ knegᵀ → Akk_raw — same blockKg (one KG GM plane).
        // Note: Catlass BlockMmad may still MTE2 B internally; c6 profiles / strengthens L1 keep.
        blockMmad(blockW, blockKg, blockAkk, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildPrefixMask(LocalTensor<float> dst, uint64_t prefix, uint64_t count)
    {
        if (prefix > count) {
            prefix = count;
        }
        Duplicate(dst, 0.0f, static_cast<uint32_t>(count));
        if (prefix > 0) {
            Duplicate(dst, 1.0f, static_cast<uint32_t>(prefix));
        }
        PipeBarrier<PIPE_V>();
    }

    // Phase B + P2: vector tril / scale / β via Brcb (no GetValue).
    // Note: BinaryRepeatParams *RepStride is in 32B blocks — for BC=16 row → stride 2 (not 8; that is BT=64).
    __aicore__ inline void ApplyTrilScaleBeta(LocalTensor<float> aqk, LocalTensor<float> akk,
                                              LocalTensor<float> beta, uint64_t valid)
    {
        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        LocalTensor<float> mask = tmpBuf_.Get<float>();
        LocalTensor<float> betaBrcb = vecBuf_.Get<float>();
        const uint8_t rowBlk = static_cast<uint8_t>((bc_ * sizeof(float)) / 32);

        Muls(aqk, aqk, scale_, elems);
        PipeBarrier<PIPE_V>();

        const uint8_t brcbRepeat = static_cast<uint8_t>((bc_ + 7) / 8);
        Brcb(betaBrcb, beta, brcbRepeat, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint64_t col = 0; col < bc_; col += 8) {
            Mul(akk[static_cast<uint32_t>(col)], akk[static_cast<uint32_t>(col)], betaBrcb, 8,
                static_cast<uint8_t>(bc_), {1, 1, 1, rowBlk, rowBlk, 1});
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t i = 0; i < bc_; ++i) {
            const uint64_t aqkPrefix = (i < valid) ? (i + 1) : 0;    // tril incl diag
            const uint64_t akkPrefix = (i < valid && i > 0) ? i : 0; // strict tril
            BuildPrefixMask(mask, aqkPrefix, bc_);
            Mul(aqk[static_cast<uint32_t>(i * bc_)], aqk[static_cast<uint32_t>(i * bc_)], mask,
                static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
            BuildPrefixMask(mask, akkPrefix, bc_);
            Mul(akk[static_cast<uint32_t>(i * bc_)], akk[static_cast<uint32_t>(i * bc_)], mask,
                static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
        }
        Muls(akk, akk, -1.0f, elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBetaRows(uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t valid,
                                        LocalTensor<float> beta)
    {
        // Always Duplicate from aligned base — never Duplicate(beta[valid], ...) (UB misalign).
        Duplicate(beta, 0.0f, static_cast<uint32_t>(bc_));
        PipeBarrier<PIPE_V>();
        if (valid == 0) {
            return;
        }
        LocalTensor<T> betaIn = inBuf_.Get<T>();
        CopyVectorIn(betaIn, beta_, BetaOff(bIdx, iHv, tok0), valid);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(beta, betaIn, RoundMode::CAST_NONE, static_cast<uint32_t>(valid));
        PipeBarrier<PIPE_V>();
    }

    // Phase A: whole-row Cast + DataCopy (no per-element SetValue / SyncSV Cast(1)).
    // Phase A helpers kept for scalar-style single-row debug; Cube Post uses PostSubStore batched path.
    __aicore__ inline void StoreAqkRow(uint64_t b, uint64_t hv, uint64_t tok, uint64_t col, LocalTensor<float> row)
    {
        LocalTensor<T> st = inBuf_.Get<T>();
        Cast(st, row, RoundMode::CAST_RINT, static_cast<uint32_t>(bc_));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(aqk_, AqkOff(b, hv, tok, col), st, bc_);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    __aicore__ inline void StoreAkkdRow(uint64_t b, uint64_t hv, uint64_t tok, LocalTensor<float> row)
    {
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(akkd_, AkkdOff(b, hv, tok), row, bc_);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // Phase C: fp32 16×16 Cube GEMM for MCH (align chunk_kda_fwd::CubeGemmSolveSub).
    __aicore__ inline void CubeGemmSolve(GlobalTensor<float> &tensorA, uint64_t baseA, GlobalTensor<float> &tensorB,
                                         uint64_t baseB, GlobalTensor<float> &tensorC, uint64_t baseC)
    {
        using ElementA = float;
        using ElementB = float;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaSolveDispatchPolicy, KdaSolveL1TileShape,
                                                              KdaSolveL0TileShape, ElementA, ElementB, ElementC, void,
                                                              TileCopy>;
        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, bc_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(bc_, bc_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);
        Catlass::GemmCoord shape{static_cast<uint32_t>(bc_), static_cast<uint32_t>(bc_),
                                 static_cast<uint32_t>(bc_)};
        auto tA = tla::MakeTensor(tensorA[baseA], layoutA, Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(tensorB[baseB], layoutB, Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(tensorC[baseC], layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        blockMmad(blockA, blockB, blockC, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AddSolveTmpToX(uint64_t slot)
    {
        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> x = arena;
        LocalTensor<float> tmp = arena[elems];
        DataCopy(x, solveWs_[SolveOff(slot, SOLVE_X, 0, 0)], elems);
        DataCopy(tmp, solveWs_[SolveOff(slot, SOLVE_TMP, 0, 0)], elems);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Add(x, x, tmp, elems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_X, 0, 0), x, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // P5: dual-AIV contiguous half-row Add (X += TMP).
    __aicore__ inline void AddSolveTmpToXRows(uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint32_t nRows = static_cast<uint32_t>(rowEnd - rowBegin);
        const uint32_t elems = nRows * static_cast<uint32_t>(bc_);
        const uint64_t xOff = SolveOff(slot, SOLVE_X, rowBegin, 0);
        const uint64_t tOff = SolveOff(slot, SOLVE_TMP, rowBegin, 0);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> x = arena;
        LocalTensor<float> tmp = arena[elems];
        DataCopy(x, solveWs_[xOff], elems);
        DataCopy(tmp, solveWs_[tOff], elems);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Add(x, x, tmp, elems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(solveWs_, xOff, x, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // Write L (cmat AKK) and X0=I-L (solve X). akk UB holds -L after ApplyTrilScaleBeta.
    // Do not touch aqkBuf_ — it still holds tril(Aqk) for the final store.
    // P2: diagonal I via aligned prefix one-hot Add (no Get/Set; no misaligned Adds(1)).
    __aicore__ inline void WriteSolveInputs(uint64_t slot, LocalTensor<float> akkNegL)
    {
        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> lMat = arena;
        LocalTensor<float> xMat = arena[elems];
        LocalTensor<float> mask = arena[2 * elems];
        LocalTensor<float> oneHot = arena[2 * elems + bc_];

        Muls(lMat, akkNegL, -1.0f, elems); // L = -(-L)
        PipeBarrier<PIPE_V>();
        Adds(xMat, akkNegL, 0.0f, elems); // X = -L
        PipeBarrier<PIPE_V>();
        for (uint64_t i = 0; i < bc_; ++i) {
            // one-hot(i) = prefix(i+1) - prefix(i); Add onto row (UB-aligned).
            BuildPrefixMask(mask, i + 1, bc_);
            BuildPrefixMask(oneHot, i, bc_);
            Sub(mask, mask, oneHot, static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
            Add(xMat[static_cast<uint32_t>(i * bc_)], xMat[static_cast<uint32_t>(i * bc_)], mask,
                static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(cmatWs_, CmatOff(slot, PLANE_AKK, 0, 0), lMat, elems);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_X, 0, 0), xMat, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    __aicore__ inline void WriteSolveInputsEmpty(uint64_t slot)
    {
        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> z = arena;
        LocalTensor<float> x = arena[elems];
        LocalTensor<float> mask = arena[2 * elems];
        LocalTensor<float> oneHot = arena[2 * elems + bc_];
        Duplicate(z, 0.0f, elems);
        Duplicate(x, 0.0f, elems);
        PipeBarrier<PIPE_V>();
        for (uint64_t i = 0; i < bc_; ++i) {
            BuildPrefixMask(mask, i + 1, bc_);
            BuildPrefixMask(oneHot, i, bc_);
            Sub(mask, mask, oneHot, static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
            Add(x[static_cast<uint32_t>(i * bc_)], x[static_cast<uint32_t>(i * bc_)], mask,
                static_cast<uint32_t>(bc_));
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(cmatWs_, CmatOff(slot, PLANE_AKK, 0, 0), z, elems);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_X, 0, 0), x, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    __aicore__ inline void ComputeMchAic(uint64_t slot)
    {
        const uint64_t lBase = CmatOff(slot, PLANE_AKK, 0, 0);
        uint64_t yBase = SolveOff(slot, SOLVE_Y0, 0, 0);
        uint64_t yNext = SolveOff(slot, SOLVE_Y1, 0, 0);
        const uint64_t xBase = SolveOff(slot, SOLVE_X, 0, 0);
        const uint64_t tmpBase = SolveOff(slot, SOLVE_TMP, 0, 0);

        Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
        // Y = L @ L
        CubeGemmSolve(cmatWs_, lBase, cmatWs_, lBase, solveWs_, yBase);
        for (uint32_t iter = 0; iter < MCH_ITERS; ++iter) {
            CubeGemmSolve(solveWs_, xBase, solveWs_, yBase, solveWs_, tmpBase);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(solveDoneFlag_);
            // P5: overlap Y@Y with AIV X+=TMP (align chunk_kda_fwd MCH schedule).
            if (iter + 1 < MCH_ITERS) {
                CubeGemmSolve(solveWs_, yBase, solveWs_, yBase, solveWs_, yNext);
                const uint64_t tmpY = yBase;
                yBase = yNext;
                yNext = tmpY;
            }
            Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
        }
    }

    // Phase D split: write L/X for MCH (AIV0 only). Leaves tril(aqk) in aqkBuf_.
    __aicore__ inline void PostSubWriteSolve(uint64_t bIdx, uint64_t iHv, uint64_t bos, uint64_t localT,
                                             uint64_t localChunk, uint64_t iSub, uint64_t slot)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        const bool empty = (iTi >= localT);
        const uint64_t valid = empty ? 0 : ((iTi + bc_ <= localT) ? bc_ : (localT - iTi));

        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        LocalTensor<float> beta = betaBuf_.Get<float>();

        if (!empty) {
            const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
            DataCopy(aqk, cmatWs_[CmatOff(slot, PLANE_AQK, 0, 0)], elems);
            DataCopy(akk, cmatWs_[CmatOff(slot, PLANE_AKK, 0, 0)], elems);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            PipeBarrier<PIPE_V>();
            LoadBetaRows(bIdx, iHv, bos + iTi, valid, beta);
            ApplyTrilScaleBeta(aqk, akk, beta, valid);
            WriteSolveInputs(slot, akk);
        } else {
            WriteSolveInputsEmpty(slot);
        }
    }

    // P5: both AIVs Add half-rows; Y@Y may overlap on AIC. Barrier before ready pulse.
    __aicore__ inline void PostSubMchAdds(uint64_t slot, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        const uint64_t rowBegin = (bc_ * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (bc_ * (subBlockIdx + 1)) / subBlockNum;
        for (uint32_t iter = 0; iter < MCH_ITERS; ++iter) {
            Catlass::Arch::CrossCoreWaitFlag(solveDoneFlag_);
            AddSolveTmpToXRows(slot, rowBegin, rowEnd);
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(solveReadyFlag_);
        }
    }

    // P4: one Cast + strided aqk DataCopyPad + contiguous akkd — no per-row HardEvent.
    __aicore__ inline void PostSubStore(uint64_t bIdx, uint64_t iHv, uint64_t bos, uint64_t localT,
                                       uint64_t localChunk, uint64_t iSub, uint64_t slot)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        if (iTi >= localT) {
            return;
        }
        const uint64_t valid = (iTi + bc_ <= localT) ? bc_ : (localT - iTi);
        if (valid == 0) {
            return;
        }
        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        const uint32_t live = static_cast<uint32_t>(valid * bc_);
        DataCopy(akk, solveWs_[SolveOff(slot, SOLVE_X, 0, 0)], elems);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> aqkT = vecBuf_.Get<T>();
        Cast(aqkT, aqk, RoundMode::CAST_RINT, live);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

        const uint64_t tok0 = bos + iTi;
        const uint64_t aqkBase = AqkOff(bIdx, iHv, tok0, iSub * bc_);
        DataCopyExtParams aqkParams;
        aqkParams.blockCount = static_cast<uint16_t>(valid);
        aqkParams.blockLen = static_cast<uint32_t>(bc_ * sizeof(T));
        aqkParams.srcStride = 0;
        aqkParams.dstStride = static_cast<uint32_t>((bt_ - bc_) * sizeof(T));
        aqkParams.rsv = 0;
        DataCopyPad(aqk_[aqkBase], aqkT, aqkParams);

        CopyVectorOut(akkd_, AkkdOff(bIdx, iHv, tok0), akk, live);

        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // Phase D: prep(0); for i: wait mmad(i) → write solve(i) → kick MCH → prep(i+1)‖MCH → store(i)
    __aicore__ inline void ProcessChunkAiv(uint64_t task)
    {
        uint64_t iB = 0, iHv = 0, iH = 0, iChunk = 0;
        DecodeTask(task, iB, iHv, iH, iChunk);
        uint64_t bos = 0, localT = t_, localChunk = iChunk, bIdx = iB;
        ResolveChunk(iChunk, iB, bos, localT, localChunk, bIdx);

        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());

        // Prologue: fill slot0 so AIC can start MMAD(0).
        {
            const uint64_t slot0 = 0 % depth_;
            PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, 0, slot0, subBlockIdx, subBlockNum);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
        }

        for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
            const uint64_t slot = iSub % depth_;
            Catlass::Arch::CrossCoreWaitFlag(doneFlag_);

            if (subBlockIdx == 0) {
                PostSubWriteSolve(bIdx, iHv, bos, localT, localChunk, iSub, slot);
            }
            // AIV1 waits until L/X are visible before pulsing solveReady.
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(solveReadyFlag_);

            // Overlap prep(i+1) with AIC MCH(i) (Y=L² / first X@Y) on the other DEPTH slot.
            if (iSub + 1 < nc_) {
                const uint64_t next = iSub + 1;
                const uint64_t slotNext = next % depth_;
                PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, next, slotNext, subBlockIdx, subBlockNum);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
            }

            PostSubMchAdds(slot, subBlockIdx, subBlockNum);
            if (subBlockIdx == 0) {
                PostSubStore(bIdx, iHv, bos, localT, localChunk, iSub, slot);
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessChunkAic(uint64_t task)
    {
        (void)task;
        for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
            const uint64_t slot = iSub % depth_;
            Catlass::Arch::CrossCoreWaitFlag(readyFlag_);
            ComputeMmad(slot);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(doneFlag_);
            ComputeMchAic(slot);
        }
    }

    GlobalTensor<T> q_, k_, g_, beta_, aqk_, scoreWs_;
    GlobalTensor<float> akkd_, cmatWs_, solveWs_;
    GlobalTensor<int64_t> cuSeqlens_, chunkIndices_;
    TPipe *pipe_ = nullptr;
    TBuf<> vecBuf_, midBuf_, betaBuf_, aqkBuf_, akkBuf_, tmpBuf_, inBuf_, i64Buf_, scalarBuf_, zeroBuf_;
    uint64_t batch_ = 0, t_ = 0, h_ = 0, hv_ = 0, kDim_ = 0, bt_ = 0, bc_ = 0, nc_ = 0, totalTasks_ = 0,
             usedCoreNum_ = 0, depth_ = SCORE_QUEUE_DEPTH, coreIdx_ = 0, group_ = 1;
    bool hasVarlen_ = false;
    float scale_ = 1.0f;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_fwd_intra_sub_chunk(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta,
                                                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk,
                                                                    GM_ADDR akkd, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        if (tilingData.dataType == 1) {
            ChunkKdaFwdIntraSubChunkScalarKernel<bfloat16_t> op;
            op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, tilingData, &pipe);
            op.Process();
        } else {
            ChunkKdaFwdIntraSubChunkScalarKernel<half> op;
            op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, tilingData, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if (tilingData.dataType == 1) {
            if ASCEND_IS_AIC {
                ChunkKdaFwdIntraSubChunkCubeKernel<bfloat16_t> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdIntraSubChunkCubeKernel<bfloat16_t> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.ProcessAiv();
            }
        } else {
            if ASCEND_IS_AIC {
                ChunkKdaFwdIntraSubChunkCubeKernel<half> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdIntraSubChunkCubeKernel<half> op;
                op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tilingData, &pipe);
                op.ProcessAiv();
            }
        }
    }
}
