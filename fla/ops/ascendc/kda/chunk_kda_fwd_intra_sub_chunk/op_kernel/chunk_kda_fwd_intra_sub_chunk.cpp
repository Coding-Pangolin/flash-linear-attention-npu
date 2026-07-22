/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — BNSD + GVA.
 * tiling key 0: AIV scalar fallback
 * tiling key 1: MIX_AIC_1_2 Cube (Score Tile dual GEMM + L0 ACC Dual MCH)
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
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

// L0 ACC MCH bring-up (see L0_ACC_MCH_DESIGN.md / PHASE_B_DUAL_PLAN.md):
//   USE_MCH_L0_GEMM=1 : CubeGemmSolve → Tile stack (L0.1 gate; S5b CV unchanged)
//   USE_MCH_L0_ACC=1  : classic Neumann + L0 ACC; single CV; Store SOLVE_X (implies GEMM path)
//   USE_MCH_L0_DUAL=1 : Phase B X∥Y dual-buffer (implies ACC)
//   USE_MCH_S2B_STEAL=1: after Dual green, steal MMAD(i+1) post solveDone (default off)
//   USE_MCH_L1_RESIDENT=1: T4 — chunk-reuse MCH Resource + I L1 residence;
//     intermediate X Fixpipe→SOLVE_TMP→L1, final X→SOLVE_X (910B has no L0C→L1)
//   USE_S2C_BATCH=1: T5 — per wave (size=depth): MMAD×n then MCH×n (fewer barriers)
// AIV S4 + Prefetch (AIV_S4_PREFETCH_PLAN.md) — 910B, no fusion:
//   USE_S4_NO_POST_BARRIER=1: drop WriteSolve→solveReady CrossCoreBarrier (0x2 Set already joins AIVs)
//   USE_SCORE_SOFT_PREFETCH=1: prefetch next KG into Score L1B during MCH wait (R2)
// WriteSolve half-load (WS_HALF_LOAD_PLAN.md):
//   USE_WS_HALF_LOAD=1: each AIV DataCopy only its row half of Aqk/Akk (drop 2× GM→UB)
//   USE_PREP_BEFORE_DONE=1: Prep(next)+ready after Store, before WaitDone(next) (W2)
// Score Tile (SCORE_TILE_CROSSCORE_PLAN.md):
//   USE_SCORE_TILE_MMAD=1 : Tile dual GEMM + L1B(kneg) residence (default on)
#ifndef USE_MCH_L0_GEMM
#define USE_MCH_L0_GEMM 1
#endif
#ifndef USE_MCH_L0_ACC
#define USE_MCH_L0_ACC 1
#endif
#ifndef USE_MCH_L0_DUAL
#define USE_MCH_L0_DUAL 1
#endif
#ifndef USE_MCH_S2B_STEAL
#define USE_MCH_S2B_STEAL 0
#endif
#ifndef USE_MCH_L1_RESIDENT
#define USE_MCH_L1_RESIDENT 1
#endif
#ifndef USE_S2C_BATCH
#define USE_S2C_BATCH 0 // T5 tried; Dur 5.36 > T4 4.11 — keep code, default off
#endif
#ifndef USE_S4_NO_POST_BARRIER
#define USE_S4_NO_POST_BARRIER 1
#endif
#ifndef USE_SCORE_SOFT_PREFETCH
#define USE_SCORE_SOFT_PREFETCH 0 // R2 tried; Dur 3.818 ≥ S4a 3.802 — keep code, default off
#endif
#ifndef USE_WS_HALF_LOAD
#define USE_WS_HALF_LOAD 0 // W1 tried; Dur 3.801 ≈ S4a 3.802 — keep code, default off
#endif
#ifndef USE_PREP_BEFORE_DONE
#define USE_PREP_BEFORE_DONE 0 // W2 tried; Prep-before-WaitDone → aqk NaN — keep code, default off
#endif
#ifndef USE_SCORE_TILE_MMAD
#define USE_SCORE_TILE_MMAD 1
#endif
#if USE_MCH_L0_DUAL
#undef USE_MCH_L0_ACC
#define USE_MCH_L0_ACC 1
#endif
#if USE_MCH_L0_ACC
#undef USE_MCH_L0_GEMM
#define USE_MCH_L0_GEMM 1
#endif
#if USE_MCH_L1_RESIDENT
#undef USE_MCH_L0_DUAL
#define USE_MCH_L0_DUAL 1
#undef USE_MCH_L0_ACC
#define USE_MCH_L0_ACC 1
#endif
#if USE_S2C_BATCH
// S2c needs Dual ACC path (single solveReady/Done per sub).
#undef USE_MCH_L0_ACC
#define USE_MCH_L0_ACC 1
#undef USE_MCH_S2B_STEAL
#define USE_MCH_S2B_STEAL 0
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
constexpr uint32_t SCORE_QUEUE_DEPTH = 2; // S2c (off) wanted 4; keep 2 with USE_S2C_BATCH=0
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

    // S3: one Duplicate + three CopyOut (QG/W/KG) — was 3× (Dup+sync+store).
    __aicore__ inline void ZeroScorePlanes(uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
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
        CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_QG, rowBegin, 0), z, elems);
        CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_W, rowBegin, 0), z, elems);
        CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_KG, rowBegin, 0), z, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // S1: float arena (6) + typed q/k/g + 3 plane outs (6×T) for batched MTE3.
    __aicore__ inline uint64_t PrepMaxTileRows() const
    {
        const uint64_t bytesPerElem = 6ull * sizeof(float) + 6ull * sizeof(T);
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

    // P1 + S1/S3: mid once per PrepareSub; zero 3 planes together; one MTE3 burst for QG/W/KG.
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
            ZeroScorePlanes(slot, rowBegin, rowEnd);
            return;
        }

        // Midpoint is per iSub (different iTi); load once before tiles (not per tile).
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
                ZeroScorePlanes(slot, tileRow + liveRows, tileRow + tileRows);
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
            LocalTensor<T> gT = typed[2 * elems];
            LocalTensor<T> outQg = typed[3 * elems];
            LocalTensor<T> outW = typed[4 * elems];
            LocalTensor<T> outKg = typed[5 * elems];

            const uint64_t tok0 = bos + iTi + tileRow;
            CopyVectorIn(qT, q_, QkRowOff(bIdx, iH, tok0), elems);
            CopyVectorIn(kT, k_, QkRowOff(bIdx, iH, tok0), elems);
            CopyVectorIn(gT, g_, HvRowOff(bIdx, iHv, tok0), elems);
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            Cast(qFp, qT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp, kT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(gFp, gT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
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

            Mul(out, qFp, expP, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outQg, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(out, kFp, expP, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outW, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(out, kFp, expN, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp16(out, static_cast<uint32_t>(elems));
            Cast(outKg, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_QG, tileRow, 0), outQg, elems);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_W, tileRow, 0), outW, elems);
            CopyVectorOut(scoreWs_, ScoreOff(slot, PLANE_KG, tileRow, 0), outKg, elems);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
            SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        }
    }

    // HARD: same blockKNeg for both MMADs (GM one KG plane).
    // USE_SCORE_TILE_MMAD: Tile stack + L1B(kneg) residence (SCORE_TILE_CROSSCORE_PLAN.md T1).
    // ElementA/B = T (bf16/fp16), ElementC = fp32 — same as chunk_kda_fwd::ComputeRawAqkAkkCubeBlock.
    __aicore__ inline void ComputeMmad(uint64_t slot, Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;

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

#if USE_SCORE_TILE_MMAD
        // Tile dual GEMM: load kneg once into L1B; swap L1A (qg → w). Events avoid MCH_EVT=2.
        // Self-contained Set/Wait pairs (same style as MchLoadGmToL1*) — no cross-call priming.
        constexpr uint16_t SCORE_EVT = 3;
        const uint32_t m = shape.m();
        const uint32_t n = shape.n();
        const uint32_t k = shape.k();
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

        // --- MMAD1: qg @ kneg → Aqk；先齐搬 B+A 再算 ---
#if USE_SCORE_SOFT_PREFETCH
        if (scoreKgPrefetchPending_ && scoreKgPrefetchSlot_ == slot) {
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            scoreKgPrefetchPending_ = false;
        } else {
            copyGmToL1B(tL1B, blockKg);
            SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        }
#else
        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
#endif
        copyGmToL1A(tL1A, blockQg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        copyL0CToGm(blockAqk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);

        // --- MMAD2: w @ kneg → Akk；L1B(kneg) 驻留，只换 L1A ---
        copyGmToL1A(tL1A, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
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
#else
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;
        BlockMmad blockMmad(resource);
        // MMAD1: qg @ knegᵀ → Aqk_raw  (loads B=kneg into L1/L0)
        blockMmad(blockQg, blockKg, blockAqk, shape);
        PipeBarrier<PIPE_ALL>();
        // MMAD2: kpos @ knegᵀ → Akk_raw — same blockKg (one KG GM plane).
        blockMmad(blockW, blockKg, blockAkk, shape);
        PipeBarrier<PIPE_ALL>();
#endif
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

    // P3: bitmask for Select — 1s zero upper triangle (same convention as chunk_kda_fwd).
    __aicore__ inline uint64_t BuildCausalMask(uint64_t threshold, uint64_t colBegin) const
    {
        if (threshold <= colBegin) {
            return ~0ULL;
        }
        if (threshold >= colBegin + bc_) {
            return 0ULL;
        }
        return ~0ULL << (threshold - colBegin);
    }

    __aicore__ inline void BuildCausalSelectMasks(LocalTensor<uint8_t> aqkMask, LocalTensor<uint8_t> akkMask,
                                                  uint64_t rowBegin, uint64_t rowCount, uint64_t valid)
    {
        __ubuf__ uint64_t *aqkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(aqkMask.GetPhyAddr());
        __ubuf__ uint64_t *akkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(akkMask.GetPhyAddr());
        for (uint32_t localRow = 0; localRow < static_cast<uint32_t>(rowCount); ++localRow) {
            const uint64_t row = rowBegin + localRow;
            const uint64_t aqkThr = (row < valid) ? (row + 1) : 0;           // tril incl diag
            const uint64_t akkThr = (row < valid && row > 0) ? row : 0;      // strict tril
            aqkMaskPtr[localRow] = BuildCausalMask(aqkThr, 0);
            akkMaskPtr[localRow] = BuildCausalMask(akkThr, 0);
        }
    }

    // P3: one Select pass for BC×rowCount (bc_≤16 → single uint64 mask / row).
    __aicore__ inline void SelectCausalRows(LocalTensor<float> aqkMat, LocalTensor<float> akkMat,
                                            uint64_t rowBegin, uint64_t rowCount, uint64_t valid)
    {
        // Layout in vecBuf_: [0 .. BC*8) betaBrcb floats; then aqk/akk uint64 masks; then zero.
        constexpr uint32_t kBetaBrcbFloats = MAX_BC * 8;
        constexpr uint32_t kMaskBytes = MAX_BC * static_cast<uint32_t>(sizeof(uint64_t));
        LocalTensor<uint8_t> aqkMask = vecBuf_.Get<uint8_t>()[kBetaBrcbFloats * sizeof(float)];
        LocalTensor<uint8_t> akkMask = vecBuf_.Get<uint8_t>()[kBetaBrcbFloats * sizeof(float) + kMaskBytes];
        LocalTensor<float> zeroLocal =
            vecBuf_.Get<float>()[(kBetaBrcbFloats * sizeof(float) + 2 * kMaskBytes) / sizeof(float)];
        Duplicate(zeroLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();

        BuildCausalSelectMasks(aqkMask, akkMask, rowBegin, rowCount, valid);
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);

        const uint32_t base = static_cast<uint32_t>(rowBegin * bc_);
        const uint8_t rowBlk = static_cast<uint8_t>((bc_ * sizeof(float)) / 32);
        BinaryRepeatParams repeatParams = {1, 0, 1, rowBlk, 0, rowBlk};
        Select(aqkMat[base], aqkMask, zeroLocal, aqkMat[base], SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(bc_), static_cast<uint8_t>(rowCount), repeatParams);
        Select(akkMat[base], akkMask, zeroLocal, akkMat[base], SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(bc_), static_cast<uint8_t>(rowCount), repeatParams);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVT_V_S);
        WaitFlag<HardEvent::V_S>(EVT_V_S);
    }

    // Phase B + P2/P3/P6: scale + Brcb β + Select tril; dual-AIV half-rows.
    // Note: BinaryRepeatParams *RepStride is in 32B blocks — for BC=16 row → stride 2 (not 8; that is BT=64).
    __aicore__ inline void ApplyTrilScaleBeta(LocalTensor<float> aqk, LocalTensor<float> akk,
                                              LocalTensor<float> beta, uint64_t valid, uint64_t rowBegin,
                                              uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint64_t rowCount = rowEnd - rowBegin;
        const uint32_t live = static_cast<uint32_t>(rowCount * bc_);
        const uint32_t base = static_cast<uint32_t>(rowBegin * bc_);
        LocalTensor<float> betaBrcb = vecBuf_.Get<float>();
        const uint8_t rowBlk = static_cast<uint8_t>((bc_ * sizeof(float)) / 32);

        Muls(aqk[base], aqk[base], scale_, live);
        PipeBarrier<PIPE_V>();

        const uint8_t brcbRepeat = static_cast<uint8_t>((rowCount + 7) / 8);
        Brcb(betaBrcb, beta[static_cast<uint32_t>(rowBegin)], brcbRepeat, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint64_t col = 0; col < bc_; col += 8) {
            Mul(akk[base + static_cast<uint32_t>(col)], akk[base + static_cast<uint32_t>(col)], betaBrcb, 8,
                static_cast<uint8_t>(rowCount), {1, 1, 1, rowBlk, rowBlk, 1});
            PipeBarrier<PIPE_V>();
        }

        SelectCausalRows(aqk, akk, rowBegin, rowCount, valid);
        Muls(akk[base], akk[base], -1.0f, live);
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

    // Phase C: fp32 16×16 GM GEMM. BlockMmad (baseline) or Tile stack (L0.1 USE_MCH_L0_GEMM).
    __aicore__ inline void CubeGemmSolve(GlobalTensor<float> &tensorA, uint64_t baseA, GlobalTensor<float> &tensorB,
                                         uint64_t baseB, GlobalTensor<float> &tensorC, uint64_t baseC)
    {
#if USE_MCH_L0_GEMM
        MchGemmSolve(tensorA, baseA, tensorB, baseB, tensorC, baseC);
#else
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
#endif
    }

    // ---- L0 ACC helpers (L0_ACC_MCH_DESIGN.md). fp32 → Catlass CopyL1ToL0 / LoadData3D only. ----
    static constexpr uint32_t MCH_L1_BYTES = MAX_BC * MAX_BC * sizeof(float);
    static constexpr uint32_t MCH_L0_BYTES = MAX_BC * MAX_BC * sizeof(float);
    // T4: keep MCH L1 past Score Tile footprint (A+B ≤ MAX_BC×MAX_K×sizeof(T) each) so I survives Score.
    static constexpr uint32_t MCH_L1_BASE =
#if USE_MCH_L1_RESIDENT
        MAX_BC * MAX_K * sizeof(T) * 2;
#else
        0;
#endif
    // Dedicated HardEvent ids — avoid clashing with Catlass BlockMmad priming on id 0/1.
    static constexpr uint16_t MCH_EVT = 2;
    static constexpr uint16_t EVT_X = 0; // Phase B X stream (after PIPE_ALL from Score)
    static constexpr uint16_t EVT_Y = 1; // Phase B Y stream

    using MchTileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, float, Catlass::layout::RowMajor, float,
                                               Catlass::layout::RowMajor, float, Catlass::layout::RowMajor>;

    __aicore__ inline void MchLoadGmToL1A(LocalTensor<float> l1, GlobalTensor<float> &gm, uint64_t base)
    {
        auto layoutGm = tla::MakeLayout<float, Catlass::layout::RowMajor>(bc_, bc_);
        auto tGm = tla::MakeTensor(gm[base], layoutGm, Catlass::Arch::PositionGM{});
        auto blockGm = GetTile(tGm, tla::MakeCoord(0, 0), tla::MakeShape(bc_, bc_));
        auto layoutL1 = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1A>(bc_, bc_);
        auto tL1 = tla::MakeTensor(l1, layoutL1, Catlass::Arch::PositionL1{});
        typename MchTileCopy::template CopyGmToL1A<decltype(blockGm)> copy;
        copy(tL1, blockGm);
        SetFlag<HardEvent::MTE2_MTE1>(MCH_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(MCH_EVT);
    }

    __aicore__ inline void MchLoadGmToL1B(LocalTensor<float> l1, GlobalTensor<float> &gm, uint64_t base)
    {
        auto layoutGm = tla::MakeLayout<float, Catlass::layout::RowMajor>(bc_, bc_);
        auto tGm = tla::MakeTensor(gm[base], layoutGm, Catlass::Arch::PositionGM{});
        auto blockGm = GetTile(tGm, tla::MakeCoord(0, 0), tla::MakeShape(bc_, bc_));
        auto layoutL1 = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1B>(bc_, bc_);
        auto tL1 = tla::MakeTensor(l1, layoutL1, Catlass::Arch::PositionL1{});
        typename MchTileCopy::template CopyGmToL1B<decltype(blockGm)> copy;
        copy(tL1, blockGm);
        SetFlag<HardEvent::MTE2_MTE1>(MCH_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(MCH_EVT);
    }

    __aicore__ inline void MchFixpipeToGm(LocalTensor<float> l0c, GlobalTensor<float> &gm, uint64_t base)
    {
        auto layoutGm = tla::MakeLayout<float, Catlass::layout::RowMajor>(bc_, bc_);
        auto tGm = tla::MakeTensor(gm[base], layoutGm, Catlass::Arch::PositionGM{});
        auto blockGm = GetTile(tGm, tla::MakeCoord(0, 0), tla::MakeShape(bc_, bc_));
        auto layoutL0C = tla::MakeLayoutL0C(bc_, bc_);
        auto tL0C = tla::MakeTensor(l0c, layoutL0C, Catlass::Arch::PositionL0C{});
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        typename MchTileCopy::template CopyL0CToDst<decltype(blockGm)> copy;
#else
        typename MchTileCopy::template CopyL0CToGm<decltype(blockGm)> copy;
#endif
        SetFlag<HardEvent::M_FIX>(MCH_EVT);
        WaitFlag<HardEvent::M_FIX>(MCH_EVT);
        copy(blockGm, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(MCH_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(MCH_EVT);
    }

    // AscendC Mmad with SolveTri-style params (cmatrixSource=false). No PipeBarrier here —
    // caller barriers (SolveTri only barriers before ACC).
    __aicore__ inline void MchMmad(LocalTensor<float> l0C, LocalTensor<float> l0A, LocalTensor<float> l0B, bool initC)
    {
        const uint32_t n = static_cast<uint32_t>(bc_);
        AscendC::MmadParams mmadParams;
        mmadParams.m = n;
        mmadParams.n = n;
        mmadParams.k = n;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmadParams);
    }

    // C = A@B0 [+ A@B1]. L1 already Nz.
    // SolveTri ACC pattern: preload B0→L0B[0], B1→L0B[TILE] BEFORE first Mmad; ACC uses
    // L0A (kept) + L0B[TILE] with NO mid-stream M_MTE1 rewrite of the same L0B slot.
    __aicore__ inline void MchMatmulL1AccFix(LocalTensor<float> l1A, LocalTensor<float> l1B0, LocalTensor<float> l1B1,
                                             LocalTensor<float> l0A, LocalTensor<float> l0B0, LocalTensor<float> l0B1,
                                             LocalTensor<float> l0C, GlobalTensor<float> &gmC, uint64_t baseC, bool doAcc)
    {
        using CopyL1ToL0A = typename MchTileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename MchTileCopy::CopyL1ToL0B;
        const uint32_t n = static_cast<uint32_t>(bc_);

        auto layoutL1A = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1A>(n, n);
        auto layoutL1B = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1B>(n, n);
        auto layoutL0A = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL0A>(n, n);
        auto layoutL0B = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL0B>(n, n);

        auto tL1A = GetTile(tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{}), tla::MakeCoord(0, 0),
                            tla::MakeShape(n, n));
        auto tL1B0 = GetTile(tla::MakeTensor(l1B0, layoutL1B, Catlass::Arch::PositionL1{}), tla::MakeCoord(0, 0),
                             tla::MakeShape(n, n));
        auto tL0A = GetTile(tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{}), tla::MakeCoord(0, 0),
                            tla::MakeShape(n, n));
        auto tL0B0 = GetTile(tla::MakeTensor(l0B0, layoutL0B, Catlass::Arch::PositionL0B{}), tla::MakeCoord(0, 0),
                             tla::MakeShape(n, n));

        CopyL1ToL0A copyA;
        CopyL1ToL0B copyB;
        copyA(tL0A, tL1A);
        copyB(tL0B0, tL1B0);
        if (doAcc) {
            auto tL1B1 = GetTile(tla::MakeTensor(l1B1, layoutL1B, Catlass::Arch::PositionL1{}), tla::MakeCoord(0, 0),
                                 tla::MakeShape(n, n));
            auto tL0B1 = GetTile(tla::MakeTensor(l0B1, layoutL0B, Catlass::Arch::PositionL0B{}), tla::MakeCoord(0, 0),
                                 tla::MakeShape(n, n));
            copyB(tL0B1, tL1B1);
        }
        SetFlag<HardEvent::MTE1_M>(MCH_EVT);
        WaitFlag<HardEvent::MTE1_M>(MCH_EVT);

        // L0C = A @ B0
        MchMmad(l0C, l0A, l0B0, true);
        if (doAcc) {
            // L0C += A @ B1  (SolveTri Mmad_ACC_Offset: same L0A, other L0B slot)
            PipeBarrier<PIPE_M>();
            MchMmad(l0C, l0A, l0B1, false);
        }
        PipeBarrier<PIPE_M>();
        MchFixpipeToGm(l0C, gmC, baseC);
    }

    // L0.1 gate: GM→GM single matmul via Tile stack (A/B separate L1 loads).
    __aicore__ inline void MchGemmSolve(GlobalTensor<float> &tensorA, uint64_t baseA, GlobalTensor<float> &tensorB,
                                        uint64_t baseB, GlobalTensor<float> &tensorC, uint64_t baseC)
    {
        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<float> l1A = resource.l1Buf.template GetBufferByByte<float>(0);
        LocalTensor<float> l1B = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BYTES);
        LocalTensor<float> l0A = resource.l0ABuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0B0 = resource.l0BBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0B1 = resource.l0BBuf.template GetBufferByByte<float>(MCH_L0_BYTES);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        MchLoadGmToL1A(l1A, tensorA, baseA);
        MchLoadGmToL1B(l1B, tensorB, baseB);
        MchMatmulL1AccFix(l1A, l1B, l1B, l0A, l0B0, l0B1, l0C, tensorC, baseC, false);
        PipeBarrier<PIPE_ALL>();
    }

    // Classic Neumann + L0 ACC. Eye at SolveOff(0, SOLVE_Y1). Output SOLVE_X.
    __aicore__ inline void MchL0Acc(uint64_t slot)
    {
        const uint64_t lBase = CmatOff(slot, PLANE_AKK, 0, 0);
        const uint64_t xBase = SolveOff(slot, SOLVE_X, 0, 0);
        const uint64_t yBase = SolveOff(slot, SOLVE_Y0, 0, 0);
        const uint64_t eyeBase = SolveOff(0, SOLVE_Y1, 0, 0);

        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<float> l1I = resource.l1Buf.template GetBufferByByte<float>(0);
        LocalTensor<float> l1X = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BYTES);
        LocalTensor<float> l1Y = resource.l1Buf.template GetBufferByByte<float>(2 * MCH_L1_BYTES);
        LocalTensor<float> l1L = resource.l1Buf.template GetBufferByByte<float>(3 * MCH_L1_BYTES);
        LocalTensor<float> l1B = resource.l1Buf.template GetBufferByByte<float>(4 * MCH_L1_BYTES);
        LocalTensor<float> l1Ib = resource.l1Buf.template GetBufferByByte<float>(5 * MCH_L1_BYTES);
        LocalTensor<float> l0A = resource.l0ABuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0B0 = resource.l0BBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0B1 = resource.l0BBuf.template GetBufferByByte<float>(MCH_L0_BYTES);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

        MchLoadGmToL1A(l1L, cmatWs_, lBase);
        MchLoadGmToL1B(l1B, cmatWs_, lBase);
        MchLoadGmToL1A(l1X, solveWs_, xBase);
        MchLoadGmToL1A(l1I, solveWs_, eyeBase);
        MchLoadGmToL1B(l1Ib, solveWs_, eyeBase);

        // Y = L @ L
        MchMatmulL1AccFix(l1L, l1B, l1B, l0A, l0B0, l0B1, l0C, solveWs_, yBase, false);
        MchLoadGmToL1A(l1Y, solveWs_, yBase);

        for (uint32_t iter = 0; iter < MCH_ITERS; ++iter) {
            MchLoadGmToL1B(l1B, solveWs_, yBase);
            // L0C = X@I; L0C += X@Y  (dual L0B preload, SolveTri-style)
            MchMatmulL1AccFix(l1X, l1Ib, l1B, l0A, l0B0, l0B1, l0C, solveWs_, xBase, true);
            if (iter + 1 < MCH_ITERS) {
                MchLoadGmToL1A(l1X, solveWs_, xBase);
                MchLoadGmToL1A(l1Y, solveWs_, yBase);
                MchLoadGmToL1B(l1B, solveWs_, yBase);
                MchMatmulL1AccFix(l1Y, l1B, l1B, l0A, l0B0, l0B1, l0C, solveWs_, yBase, false);
                MchLoadGmToL1A(l1Y, solveWs_, yBase);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Phase B: SolveTri-style X∥Y dual-buffer (PHASE_B_DUAL_PLAN.md). No MBH.
    __aicore__ inline void MchCopyL1AToL0A(LocalTensor<float> l0A, LocalTensor<float> l1A)
    {
        using CopyL1ToL0A = typename MchTileCopy::CopyL1ToL0A;
        const uint32_t n = static_cast<uint32_t>(bc_);
        auto layoutL1A = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1A>(n, n);
        auto layoutL0A = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL0A>(n, n);
        auto tL1 = GetTile(tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{}), tla::MakeCoord(0, 0),
                           tla::MakeShape(n, n));
        auto tL0 = GetTile(tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{}), tla::MakeCoord(0, 0),
                           tla::MakeShape(n, n));
        CopyL1ToL0A copy;
        copy(tL0, tL1);
    }

    __aicore__ inline void MchCopyL1BToL0B(LocalTensor<float> l0B, LocalTensor<float> l1B)
    {
        using CopyL1ToL0B = typename MchTileCopy::CopyL1ToL0B;
        const uint32_t n = static_cast<uint32_t>(bc_);
        auto layoutL1B = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL1B>(n, n);
        auto layoutL0B = tla::MakeLayout<float, typename MchTileCopy::LayoutTagL0B>(n, n);
        auto tL1 = GetTile(tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{}), tla::MakeCoord(0, 0),
                           tla::MakeShape(n, n));
        auto tL0 = GetTile(tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{}), tla::MakeCoord(0, 0),
                           tla::MakeShape(n, n));
        CopyL1ToL0B copy;
        copy(tL0, tL1);
    }

    __aicore__ inline void MchFixpipeToGmEvt(LocalTensor<float> l0c, GlobalTensor<float> &gm, uint64_t base,
                                             uint16_t evt)
    {
        auto layoutGm = tla::MakeLayout<float, Catlass::layout::RowMajor>(bc_, bc_);
        auto tGm = tla::MakeTensor(gm[base], layoutGm, Catlass::Arch::PositionGM{});
        auto blockGm = GetTile(tGm, tla::MakeCoord(0, 0), tla::MakeShape(bc_, bc_));
        auto layoutL0C = tla::MakeLayoutL0C(bc_, bc_);
        auto tL0C = tla::MakeTensor(l0c, layoutL0C, Catlass::Arch::PositionL0C{});
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        typename MchTileCopy::template CopyL0CToDst<decltype(blockGm)> copy;
#else
        typename MchTileCopy::template CopyL0CToGm<decltype(blockGm)> copy;
#endif
        SetFlag<HardEvent::M_FIX>(evt);
        WaitFlag<HardEvent::M_FIX>(evt);
        copy(blockGm, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(evt);
        WaitFlag<HardEvent::FIX_MTE2>(evt);
    }

    // T4 USE_MCH_L1_RESIDENT: shared Resource + optional I reload; intermediate X→TMP.
    __aicore__ inline void MchL0AccDual(uint64_t slot, Catlass::Arch::Resource<KdaArchTag> &resource, bool loadEye)
    {
        const uint64_t lBase = CmatOff(slot, PLANE_AKK, 0, 0);
        const uint64_t xBase = SolveOff(slot, SOLVE_X, 0, 0);
        const uint64_t yBase = SolveOff(slot, SOLVE_Y0, 0, 0);
        const uint64_t tmpBase = SolveOff(slot, SOLVE_TMP, 0, 0);
        const uint64_t eyeBase = SolveOff(0, SOLVE_Y1, 0, 0);
#if USE_MCH_L1_RESIDENT
        const uint64_t xScratch = tmpBase; // intermediate X Fixpipe/Nd2Nz; final → xBase
#else
        const uint64_t xScratch = xBase;
        (void)tmpBase;
#endif

        LocalTensor<float> l1I = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 0);
        LocalTensor<float> l1X = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + MCH_L1_BYTES);
        LocalTensor<float> l1Y = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 2 * MCH_L1_BYTES);
        LocalTensor<float> l1L = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 3 * MCH_L1_BYTES);
        LocalTensor<float> l1Lb = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 4 * MCH_L1_BYTES);
        LocalTensor<float> l1Ib = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 5 * MCH_L1_BYTES);
        LocalTensor<float> l1Yb = resource.l1Buf.template GetBufferByByte<float>(MCH_L1_BASE + 6 * MCH_L1_BYTES);

        LocalTensor<float> l0Ax = resource.l0ABuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0Ay = resource.l0ABuf.template GetBufferByByte<float>(MCH_L0_BYTES);
        LocalTensor<float> l0Bx = resource.l0BBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0By = resource.l0BBuf.template GetBufferByByte<float>(MCH_L0_BYTES);
        LocalTensor<float> l0Cx = resource.l0CBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> l0Cy = resource.l0CBuf.template GetBufferByByte<float>(MCH_L0_BYTES);

        // Isolate from Score BlockMmad HardEvent state.
        PipeBarrier<PIPE_ALL>();

        MchLoadGmToL1A(l1L, cmatWs_, lBase);
        MchLoadGmToL1B(l1Lb, cmatWs_, lBase);
        MchLoadGmToL1A(l1X, solveWs_, xBase);
        if (loadEye) {
            MchLoadGmToL1A(l1I, solveWs_, eyeBase);
            MchLoadGmToL1B(l1Ib, solveWs_, eyeBase);
        }

        // Y0 = L@L → L1_Y (+ L1B copy)
        MchMatmulL1AccFix(l1L, l1Lb, l1Lb, l0Ax, l0Bx, l0By, l0Cx, solveWs_, yBase, false);
        MchLoadGmToL1A(l1Y, solveWs_, yBase);
        MchLoadGmToL1B(l1Yb, solveWs_, yBase);

        // Priming so first-iter Waits pass (SolveTri MCHInvertDiagonal).
        SetFlag<HardEvent::M_MTE1>(EVT_X);
        SetFlag<HardEvent::M_MTE1>(EVT_Y);
        SetFlag<HardEvent::FIX_M>(EVT_X);
        SetFlag<HardEvent::FIX_M>(EVT_Y);

        for (uint32_t iter = 0; iter < MCH_ITERS; ++iter) {
            WaitFlag<HardEvent::M_MTE1>(EVT_X);
            MchCopyL1AToL0A(l0Ax, l1X);
            MchCopyL1BToL0B(l0Bx, l1Ib);
            SetFlag<HardEvent::MTE1_M>(EVT_X);

            WaitFlag<HardEvent::M_MTE1>(EVT_Y);
            MchCopyL1AToL0A(l0Ay, l1Y);
            MchCopyL1BToL0B(l0By, l1Yb);
            SetFlag<HardEvent::MTE1_M>(EVT_Y);

            WaitFlag<HardEvent::FIX_M>(EVT_X);
            WaitFlag<HardEvent::MTE1_M>(EVT_X);
            MchMmad(l0Cx, l0Ax, l0Bx, true); // L0C_X = X@I

            if (iter + 1 < MCH_ITERS) {
                WaitFlag<HardEvent::FIX_M>(EVT_Y);
                WaitFlag<HardEvent::MTE1_M>(EVT_Y);
                MchMmad(l0Cy, l0Ay, l0By, true); // L0C_Y = Y@Y
                MchFixpipeToGmEvt(l0Cy, solveWs_, yBase, EVT_Y);
                MchLoadGmToL1A(l1Y, solveWs_, yBase);
                MchLoadGmToL1B(l1Yb, solveWs_, yBase);
                SetFlag<HardEvent::FIX_M>(EVT_Y);
            }

            PipeBarrier<PIPE_M>();
            if (iter + 1 >= MCH_ITERS) {
                WaitFlag<HardEvent::MTE1_M>(EVT_Y);
            }
            // L0C_X += X@Y; L0B[Y] still holds this-iter Y (pre-square).
            MchMmad(l0Cx, l0Ax, l0By, false);
            if (iter + 1 < MCH_ITERS) {
                MchFixpipeToGmEvt(l0Cx, solveWs_, xScratch, EVT_X);
                MchLoadGmToL1A(l1X, solveWs_, xScratch);
            } else {
                MchFixpipeToGmEvt(l0Cx, solveWs_, xBase, EVT_X);
            }

            SetFlag<HardEvent::M_MTE1>(EVT_X);
            SetFlag<HardEvent::M_MTE1>(EVT_Y);
            SetFlag<HardEvent::FIX_M>(EVT_X);
        }

        // Drain unpaired priming/loop flags (SolveTri).
        WaitFlag<HardEvent::M_MTE1>(EVT_X);
        WaitFlag<HardEvent::M_MTE1>(EVT_Y);
        WaitFlag<HardEvent::FIX_M>(EVT_X);
        WaitFlag<HardEvent::FIX_M>(EVT_Y);
        PipeBarrier<PIPE_ALL>();
    }

#if USE_SCORE_SOFT_PREFETCH
    // Issue KG→L1B into Score L1 prefix (below MCH_L1_BASE). Wait in next ComputeMmad.
    __aicore__ inline void IssuePrefetchScoreKg(Catlass::Arch::Resource<KdaArchTag> &resource, uint64_t slot)
    {
        constexpr uint16_t SCORE_EVT = 3;
        using ElementA = T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, float, LayoutTagC>;
        const uint32_t m = static_cast<uint32_t>(bc_);
        const uint32_t n = static_cast<uint32_t>(bc_);
        const uint32_t k = static_cast<uint32_t>(kDim_);
        const uint32_t l1ABytes = m * k * sizeof(ElementA);

        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        auto tensorKg = tla::MakeTensor(scoreWs_[ScoreOff(slot, PLANE_KG, 0, 0)], layoutB, Catlass::Arch::PositionGM{});
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(l1ABytes);
        auto layoutL1B = tla::MakeLayout<ElementB, typename TileCopy::LayoutTagL1B>(k, n);
        auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1B<decltype(blockKg)> copyGmToL1B;
        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        scoreKgPrefetchSlot_ = slot;
        scoreKgPrefetchPending_ = true;
    }
#endif

    __aicore__ inline void MchL0AccDual(uint64_t slot)
    {
        Catlass::Arch::Resource<KdaArchTag> resource;
        MchL0AccDual(slot, resource, true);
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

    // P3b: build I[rowBegin:rowEnd] via Select (diag bitmask), avoid prefix×2 per row.
    // Masks in tmpBuf_ (fits dual-AIV half-rows); ones in midBuf_ — keep clear of vecBuf_ arena.
    __aicore__ inline void FillIdentityRows(LocalTensor<float> eye, uint64_t rowBegin, uint64_t rowCount)
    {
        const uint32_t live = static_cast<uint32_t>(rowCount * bc_);
        LocalTensor<uint8_t> diagMask = tmpBuf_.Get<uint8_t>();
        LocalTensor<float> ones = midBuf_.Get<float>();
        Duplicate(eye, 0.0f, live);
        Duplicate(ones, 1.0f, 8);
        PipeBarrier<PIPE_V>();

        __ubuf__ uint64_t *maskPtr = reinterpret_cast<__ubuf__ uint64_t *>(diagMask.GetPhyAddr());
        for (uint32_t local = 0; local < static_cast<uint32_t>(rowCount); ++local) {
            const uint64_t i = rowBegin + local;
            // Select: mask bit1 → ones, bit0 → eye(0). Put 1 only on diagonal column i.
            maskPtr[local] = (i < 64) ? (1ULL << i) : 0ULL;
        }
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);

        const uint8_t rowBlk = static_cast<uint8_t>((bc_ * sizeof(float)) / 32);
        BinaryRepeatParams repeatParams = {1, 0, 1, rowBlk, 0, rowBlk};
        Select(eye, diagMask, ones, eye, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(bc_),
               static_cast<uint8_t>(rowCount), repeatParams);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVT_V_S);
        WaitFlag<HardEvent::V_S>(EVT_V_S);
    }

    // Write L (cmat AKK) and X0=I-L (solve X). akk UB holds -L after ApplyTrilScaleBeta.
    // Do not touch aqkBuf_ — it still holds tril(Aqk) for the final store.
    // P2/P3b/P6: X=-L+I via FillIdentityRows; row-split for dual AIV.
    __aicore__ inline void WriteSolveInputs(uint64_t slot, LocalTensor<float> akkNegL, uint64_t rowBegin,
                                            uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint64_t rowCount = rowEnd - rowBegin;
        const uint32_t live = static_cast<uint32_t>(rowCount * bc_);
        const uint32_t srcBase = static_cast<uint32_t>(rowBegin * bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> lMat = arena;
        LocalTensor<float> xMat = arena[live];
        LocalTensor<float> eye = arena[2 * live];

        Muls(lMat, akkNegL[srcBase], -1.0f, live); // L = -(-L)
        PipeBarrier<PIPE_V>();
        Adds(xMat, akkNegL[srcBase], 0.0f, live); // X = -L
        PipeBarrier<PIPE_V>();
        FillIdentityRows(eye, rowBegin, rowCount);
        Add(xMat, xMat, eye, live); // X = I - L
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(cmatWs_, CmatOff(slot, PLANE_AKK, rowBegin, 0), lMat, live);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_X, rowBegin, 0), xMat, live);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    __aicore__ inline void WriteSolveInputsEmpty(uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint64_t rowCount = rowEnd - rowBegin;
        const uint32_t live = static_cast<uint32_t>(rowCount * bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> z = arena;
        LocalTensor<float> x = arena[live];
        Duplicate(z, 0.0f, live);
        FillIdentityRows(x, rowBegin, rowCount); // X = I
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(cmatWs_, CmatOff(slot, PLANE_AKK, rowBegin, 0), z, live);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_X, rowBegin, 0), x, live);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // MCH: L0 ACC Neumann (USE_MCH_L0_ACC) or S5b closed-form.
    __aicore__ inline bool ComputeMchAic(uint64_t slot, uint64_t iSub, Catlass::Arch::Resource<KdaArchTag> &mchResource,
                                         bool loadEye)
    {
#if USE_MCH_L0_ACC
        Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
#if USE_MCH_L0_DUAL
        MchL0AccDual(slot, mchResource, loadEye);
#else
        (void)mchResource;
        (void)loadEye;
        MchL0Acc(slot);
#endif
#if USE_SCORE_SOFT_PREFETCH
        // Prep(next)+ready already posted before AIV WaitMch; hide KG Nd2Nz under Store.
        if (iSub + 1 < nc_) {
            IssuePrefetchScoreKg(mchResource, (iSub + 1) % depth_);
        }
#else
        (void)iSub;
#endif
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(solveDoneFlag_);
#if USE_MCH_S2B_STEAL
        if (iSub + 1 < nc_) {
            const uint64_t nextSlot = (iSub + 1) % depth_;
            Catlass::Arch::CrossCoreWaitFlag(readyFlag_);
            ComputeMmad(nextSlot);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(doneFlag_);
            return true;
        }
#endif
        return false;
#else
        (void)mchResource;
        (void)loadEye;
        const uint64_t lBase = CmatOff(slot, PLANE_AKK, 0, 0);
        const uint64_t xBase = SolveOff(slot, SOLVE_X, 0, 0);
        const uint64_t y0Base = SolveOff(slot, SOLVE_Y0, 0, 0);
        const uint64_t y1Base = SolveOff(slot, SOLVE_Y1, 0, 0);
        const uint64_t y2Base = SolveOff(slot, SOLVE_TMP, 0, 0);

        Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
        CubeGemmSolve(cmatWs_, lBase, cmatWs_, lBase, solveWs_, y0Base);
        CubeGemmSolve(solveWs_, y0Base, solveWs_, y0Base, solveWs_, y1Base);
        CubeGemmSolve(solveWs_, y1Base, solveWs_, y1Base, solveWs_, y2Base);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(solveDoneFlag_);

        bool stoleNext = false;
        if (iSub + 1 < nc_) {
            const uint64_t nextSlot = (iSub + 1) % depth_;
            Catlass::Arch::CrossCoreWaitFlag(readyFlag_);
            ComputeMmad(nextSlot);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(doneFlag_);
            stoleNext = true;
        }

        Catlass::Arch::CrossCoreWaitFlag(solveReadyFlag_);
        CubeGemmSolve(solveWs_, y0Base, solveWs_, y1Base, cmatWs_, lBase);
        CubeGemmSolve(cmatWs_, lBase, solveWs_, y2Base, solveWs_, y0Base);
        CubeGemmSolve(solveWs_, xBase, solveWs_, y0Base, solveWs_, y2Base);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(solveDoneFlag_);
        return stoleNext;
#endif
    }

    // S5b: Pk = I + Yk (unused when USE_MCH_L0_ACC).
    __aicore__ inline void AddIdentityToMchPlanes(uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint32_t nRows = static_cast<uint32_t>(rowEnd - rowBegin);
        const uint32_t elems = nRows * static_cast<uint32_t>(bc_);
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> y0 = arena;
        LocalTensor<float> y1 = arena[elems];
        LocalTensor<float> y2 = arena[2 * elems];
        LocalTensor<float> eye = arena[3 * elems];

        DataCopy(y0, solveWs_[SolveOff(slot, SOLVE_Y0, rowBegin, 0)], elems);
        DataCopy(y1, solveWs_[SolveOff(slot, SOLVE_Y1, rowBegin, 0)], elems);
        DataCopy(y2, solveWs_[SolveOff(slot, SOLVE_TMP, rowBegin, 0)], elems);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        FillIdentityRows(eye, rowBegin, nRows);
        Add(y0, y0, eye, elems);
        Add(y1, y1, eye, elems);
        Add(y2, y2, eye, elems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_Y0, rowBegin, 0), y0, elems);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_Y1, rowBegin, 0), y1, elems);
        CopyVectorOut(solveWs_, SolveOff(slot, SOLVE_TMP, rowBegin, 0), y2, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

#if USE_MCH_L0_ACC
    __aicore__ inline void PostSubMchWait(uint64_t slot, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        (void)slot;
        (void)subBlockIdx;
        (void)subBlockNum;
        Catlass::Arch::CrossCoreWaitFlag(solveDoneFlag_);
    }
#else
    // S5b: wait Y-powers → batched I+Y → kick product → wait X_new in TMP.
    __aicore__ inline void PostSubMchClosedForm(uint64_t slot, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        const uint64_t rowBegin = (bc_ * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (bc_ * (subBlockIdx + 1)) / subBlockNum;

        Catlass::Arch::CrossCoreWaitFlag(solveDoneFlag_);
        AddIdentityToMchPlanes(slot, rowBegin, rowEnd);
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(solveReadyFlag_);

        Catlass::Arch::CrossCoreWaitFlag(solveDoneFlag_);
    }
#endif

    // Phase D + P6: both AIVs write L/X row halves. Leaves tril(aqk) rows in aqkBuf_.
    // USE_S2C_BATCH: also spill tril(aqk) → cmat AQK so Store can reload after later WriteSolves.
    __aicore__ inline void PostSubWriteSolve(uint64_t bIdx, uint64_t iHv, uint64_t bos, uint64_t localT,
                                             uint64_t localChunk, uint64_t iSub, uint64_t slot,
                                             uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        const bool empty = (iTi >= localT);
        const uint64_t valid = empty ? 0 : ((iTi + bc_ <= localT) ? bc_ : (localT - iTi));
        const uint64_t rowBegin = (bc_ * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (bc_ * (subBlockIdx + 1)) / subBlockNum;

        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        LocalTensor<float> beta = betaBuf_.Get<float>();

        if (!empty && valid > 0) {
#if USE_WS_HALF_LOAD
            // Each AIV only needs its half-rows for tril/write; avoid 2× full 16×16 GM→UB.
            if (rowBegin < rowEnd) {
                const uint32_t halfElems = static_cast<uint32_t>((rowEnd - rowBegin) * bc_);
                const uint32_t base = static_cast<uint32_t>(rowBegin * bc_);
                DataCopy(aqk[base], cmatWs_[CmatOff(slot, PLANE_AQK, rowBegin, 0)], halfElems);
                DataCopy(akk[base], cmatWs_[CmatOff(slot, PLANE_AKK, rowBegin, 0)], halfElems);
            }
#else
            const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
            DataCopy(aqk, cmatWs_[CmatOff(slot, PLANE_AQK, 0, 0)], elems);
            DataCopy(akk, cmatWs_[CmatOff(slot, PLANE_AKK, 0, 0)], elems);
#endif
            SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
            PipeBarrier<PIPE_V>();
            LoadBetaRows(bIdx, iHv, bos + iTi, valid, beta);
            ApplyTrilScaleBeta(aqk, akk, beta, valid, rowBegin, rowEnd);
            WriteSolveInputs(slot, akk, rowBegin, rowEnd);
#if USE_S2C_BATCH
            if (rowBegin < rowEnd) {
                const uint32_t live = static_cast<uint32_t>((rowEnd - rowBegin) * bc_);
                const uint32_t srcBase = static_cast<uint32_t>(rowBegin * bc_);
                LocalTensor<float> aqkRows = aqk[srcBase];
                SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                CopyVectorOut(cmatWs_, CmatOff(slot, PLANE_AQK, rowBegin, 0), aqkRows, live);
                SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
                WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            }
#endif
        } else {
            WriteSolveInputsEmpty(slot, rowBegin, rowEnd);
        }
    }

    // Legacy P5 Add path (replaced by S5 PostSubMchClosedForm). Kept out of hot path.
    __aicore__ inline void PostSubMchAdds(uint64_t slot, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        (void)slot;
        (void)subBlockIdx;
        (void)subBlockNum;
    }

    // P4+P6: dual-AIV store of contiguous half valid-rows.
    __aicore__ inline void PostSubStore(uint64_t bIdx, uint64_t iHv, uint64_t bos, uint64_t localT,
                                       uint64_t localChunk, uint64_t iSub, uint64_t slot, uint64_t subBlockIdx,
                                       uint64_t subBlockNum)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        if (iTi >= localT) {
            return;
        }
        const uint64_t valid = (iTi + bc_ <= localT) ? bc_ : (localT - iTi);
        if (valid == 0) {
            return;
        }
        const uint64_t splitBegin = (bc_ * subBlockIdx) / subBlockNum;
        const uint64_t splitEnd = (bc_ * (subBlockIdx + 1)) / subBlockNum;
        uint64_t rowBegin = splitBegin;
        uint64_t rowEnd = splitEnd;
        if (rowBegin >= valid) {
            return;
        }
        if (rowEnd > valid) {
            rowEnd = valid;
        }
        if (rowBegin >= rowEnd) {
            return;
        }

        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        const uint32_t live = static_cast<uint32_t>((rowEnd - rowBegin) * bc_);
        const uint32_t srcBase = static_cast<uint32_t>(rowBegin * bc_);
        LocalTensor<float> akkRows = akk[srcBase];
        LocalTensor<float> aqkRows = aqk[srcBase];
#if USE_S2C_BATCH
        // tril(aqk) was spilled to cmat AQK in WriteSolve (aqkBuf_ reused across wave).
        DataCopy(aqkRows, cmatWs_[CmatOff(slot, PLANE_AQK, rowBegin, 0)], live);
#endif
#if USE_MCH_L0_ACC
        // L0 ACC: X_new in SOLVE_X
        DataCopy(akkRows, solveWs_[SolveOff(slot, SOLVE_X, rowBegin, 0)], live);
#else
        // S5b: X_new lives in SOLVE_TMP after closed-form product.
        DataCopy(akkRows, solveWs_[SolveOff(slot, SOLVE_TMP, rowBegin, 0)], live);
#endif
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> aqkT = vecBuf_.Get<T>();
        Cast(aqkT, aqkRows, RoundMode::CAST_RINT, live);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

        const uint64_t tok0 = bos + iTi + rowBegin;
        const uint64_t aqkBase = AqkOff(bIdx, iHv, tok0, iSub * bc_);
        DataCopyExtParams aqkParams;
        aqkParams.blockCount = static_cast<uint16_t>(rowEnd - rowBegin);
        aqkParams.blockLen = static_cast<uint32_t>(bc_ * sizeof(T));
        aqkParams.srcStride = 0;
        aqkParams.dstStride = static_cast<uint32_t>((bt_ - bc_) * sizeof(T));
        aqkParams.rsv = 0;
        DataCopyPad(aqk_[aqkBase], aqkT, aqkParams);

        CopyVectorOut(akkd_, AkkdOff(bIdx, iHv, tok0), akkRows, live);

        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // Phase D: prep(0); for i: wait mmad(i) → write solve(i) → kick MCH → prep(i+1)‖MCH → store(i)
    // T5 USE_S2C_BATCH: wave of depth — MMAD×n then MCH×n (one barrier per wave).
    __aicore__ inline void ProcessChunkAiv(uint64_t task)
    {
        uint64_t iB = 0, iHv = 0, iH = 0, iChunk = 0;
        DecodeTask(task, iB, iHv, iH, iChunk);
        uint64_t bos = 0, localT = t_, localChunk = iChunk, bIdx = iB;
        ResolveChunk(iChunk, iB, bos, localT, localChunk, bIdx);

        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());

        // Prologue: Prep(0)+ready ASAP (T2: Identity does not block Cube start).
        {
            const uint64_t slot0 = 0 % depth_;
            PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, 0, slot0, subBlockIdx, subBlockNum);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
#if USE_MCH_L0_ACC
            if (subBlockIdx == 0) {
                const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
                LocalTensor<float> eye = vecBuf_.Get<float>();
                FillIdentityRows(eye, 0, bc_);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
                CopyVectorOut(solveWs_, SolveOff(0, SOLVE_Y1, 0, 0), eye, elems);
                SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
                SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
                WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
#endif
        }

#if USE_S2C_BATCH
        for (uint64_t wave = 0; wave < nc_; wave += depth_) {
            const uint64_t n = (wave + depth_ <= nc_) ? depth_ : (nc_ - wave);
            // Score phase: WaitDone → WriteSolve; Prep next within wave.
            for (uint64_t j = 0; j < n; ++j) {
                const uint64_t iSub = wave + j;
                const uint64_t slot = iSub % depth_;
                Catlass::Arch::CrossCoreWaitFlag(doneFlag_);
                PostSubWriteSolve(bIdx, iHv, bos, localT, localChunk, iSub, slot, subBlockIdx, subBlockNum);
                if (j + 1 < n) {
                    const uint64_t next = iSub + 1;
                    const uint64_t slotNext = next % depth_;
                    PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, next, slotNext, subBlockIdx, subBlockNum);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
                }
            }
            // One dual-AIV barrier for the whole wave's WriteSolve (was N barriers).
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
            // MCH + Store: CrossCoreFlag is 1:1 Set/Wait (not a counting semaphore).
            for (uint64_t j = 0; j < n; ++j) {
                const uint64_t iSub = wave + j;
                const uint64_t slot = iSub % depth_;
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(solveReadyFlag_);
#if USE_MCH_L0_ACC
                PostSubMchWait(slot, subBlockIdx, subBlockNum);
#else
                PostSubMchClosedForm(slot, subBlockIdx, subBlockNum);
#endif
                PostSubStore(bIdx, iHv, bos, localT, localChunk, iSub, slot, subBlockIdx, subBlockNum);
            }
            // Kick Score of next wave after slots are free.
            if (wave + n < nc_) {
                const uint64_t next = wave + n;
                const uint64_t slotNext = next % depth_;
                PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, next, slotNext, subBlockIdx, subBlockNum);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
            }
        }
#else
        for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
            const uint64_t slot = iSub % depth_;
#if USE_PREP_BEFORE_DONE
            // Prep(next) while AIC is in MMAD(i). Do NOT Set ready yet — early ready lets
            // AIC start MMAD(next) before Store(i)/MCH handshake completes (NaN regression).
            if (iSub + 1 < nc_) {
                const uint64_t next = iSub + 1;
                const uint64_t slotNext = next % depth_;
                PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, next, slotNext, subBlockIdx, subBlockNum);
            }
#endif
            Catlass::Arch::CrossCoreWaitFlag(doneFlag_);

            PostSubWriteSolve(bIdx, iHv, bos, localT, localChunk, iSub, slot, subBlockIdx, subBlockNum);
#if !USE_S4_NO_POST_BARRIER
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(solveReadyFlag_);

#if USE_PREP_BEFORE_DONE
            // Prep already done; Set ready immediately so AIC MCH→Wait ready sees no Prep gap.
            if (iSub + 1 < nc_) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
            }
#else
            if (iSub + 1 < nc_) {
                const uint64_t next = iSub + 1;
                const uint64_t slotNext = next % depth_;
                PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, next, slotNext, subBlockIdx, subBlockNum);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(readyFlag_);
            }
#endif

#if USE_MCH_L0_ACC
            PostSubMchWait(slot, subBlockIdx, subBlockNum);
#else
            PostSubMchClosedForm(slot, subBlockIdx, subBlockNum);
#endif
            PostSubStore(bIdx, iHv, bos, localT, localChunk, iSub, slot, subBlockIdx, subBlockNum);
        }
#endif
    }

    __aicore__ inline void ProcessChunkAic(uint64_t task)
    {
        (void)task;
#if USE_MCH_L1_RESIDENT
        Catlass::Arch::Resource<KdaArchTag> sharedResource;
        bool loadEye = true;
#else
        Catlass::Arch::Resource<KdaArchTag> scoreResource;
#endif

#if USE_S2C_BATCH
        for (uint64_t wave = 0; wave < nc_; wave += depth_) {
            const uint64_t n = (wave + depth_ <= nc_) ? depth_ : (nc_ - wave);
            for (uint64_t j = 0; j < n; ++j) {
                const uint64_t iSub = wave + j;
                const uint64_t slot = iSub % depth_;
                Catlass::Arch::CrossCoreWaitFlag(readyFlag_);
#if USE_MCH_L1_RESIDENT
                ComputeMmad(slot, sharedResource);
#else
                ComputeMmad(slot, scoreResource);
#endif
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(doneFlag_);
            }
            for (uint64_t j = 0; j < n; ++j) {
                const uint64_t iSub = wave + j;
                const uint64_t slot = iSub % depth_;
#if USE_MCH_L1_RESIDENT
                (void)ComputeMchAic(slot, iSub, sharedResource, loadEye);
                loadEye = false;
#else
                Catlass::Arch::Resource<KdaArchTag> mchResource;
                (void)ComputeMchAic(slot, iSub, mchResource, true);
#endif
            }
        }
#else
        bool skipMmad = false;
        for (uint64_t iSub = 0; iSub < nc_; ++iSub) {
            const uint64_t slot = iSub % depth_;
            if (!skipMmad) {
                Catlass::Arch::CrossCoreWaitFlag(readyFlag_);
#if USE_MCH_L1_RESIDENT
                ComputeMmad(slot, sharedResource);
#else
                ComputeMmad(slot, scoreResource);
#endif
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(doneFlag_);
            }
#if USE_MCH_L1_RESIDENT
            skipMmad = ComputeMchAic(slot, iSub, sharedResource, loadEye);
            loadEye = false;
#else
            Catlass::Arch::Resource<KdaArchTag> mchResource;
            skipMmad = ComputeMchAic(slot, iSub, mchResource, true);
#endif
        }
#endif
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
#if USE_SCORE_SOFT_PREFETCH
    uint64_t scoreKgPrefetchSlot_ = 0;
    bool scoreKgPrefetchPending_ = false;
#endif
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
