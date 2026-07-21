/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Aligns with GPU Triton chunk_kda_fwd_kernel_intra_sub_chunk.
 * Layout: BNSD. q/k use H; g/beta/Aqk/Akkd use HV (GVA: HV>=H, HV%H==0).
 * Head map: i_h = i_hv / (HV/H).
 *
 * Note: UB SetValue/GetValue are used for the BC×BC scalar math. Results are
 * written to GM via GlobalTensor::SetValue (not DataCopy), because SetValue
 * logical layout does not match DataCopy physical UB banks on AIV.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float LN2 = 0.6931471805599453f;
constexpr float EXP_INPUT_MAX = 10.0f;
constexpr float EXP_INPUT_MIN = -10.0f;
constexpr uint32_t MAX_K = 256;
constexpr uint32_t MAX_BC = 16;
constexpr uint32_t EVT_MTE2_V = 0;
constexpr uint32_t EVT_V_S = 4;
constexpr uint32_t EVT_S_V = 5;

template <typename T>
class ChunkKdaFwdIntraSubChunkKernel {
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
            ProcessTask(task);
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

    __aicore__ inline void StoreAqkRow(uint64_t b, uint64_t h, uint64_t t, uint64_t col, LocalTensor<float> row)
    {
        // Element-wise GM write: cast float→T via 1-element Cast (no host-side bf16 cast).
        const uint64_t base = AqkOff(b, h, t, col);
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

    __aicore__ inline void StoreAkkdRow(uint64_t b, uint64_t h, uint64_t t, LocalTensor<float> row)
    {
        const uint64_t base = AkkdOff(b, h, t);
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

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        const uint64_t bhv = batch_ * hv_;
        const uint64_t iBhv = task % bhv;
        const uint64_t rem = task / bhv;
        const uint64_t iSub = rem % nc_;
        const uint64_t iChunk = rem / nc_;
        const uint64_t iB = iBhv / hv_;
        const uint64_t iHv = iBhv % hv_;
        const uint64_t iH = iHv / group_;

        uint64_t bos = 0;
        uint64_t localT = t_;
        uint64_t localChunk = iChunk;
        uint64_t bIdx = iB;
        if (hasVarlen_) {
            const int64_t seqId = LoadI64(chunkIndices_, iChunk * 2);
            localChunk = static_cast<uint64_t>(LoadI64(chunkIndices_, iChunk * 2 + 1));
            bos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId)));
            const uint64_t eos = static_cast<uint64_t>(LoadI64(cuSeqlens_, static_cast<uint64_t>(seqId) + 1));
            localT = eos - bos;
            bIdx = 0;
        }

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

        // M = -L
        for (uint64_t i = 0; i < valid; ++i) {
            for (uint64_t j = 0; j < i; ++j) {
                const uint32_t idx = static_cast<uint32_t>(i * bc_ + j);
                akk.SetValue(idx, -akk.GetValue(idx));
            }
        }
        // Triton forward substitution
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
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_fwd_intra_sub_chunk(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta,
                                                                    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk,
                                                                    GM_ADDR akkd, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (tilingData.dataType == 1) {
        ChunkKdaFwdIntraSubChunkKernel<bfloat16_t> op;
        op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, tilingData, &pipe);
        op.Process();
    } else {
        ChunkKdaFwdIntraSubChunkKernel<half> op;
        op.Init(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, tilingData, &pipe);
        op.Process();
    }
}
