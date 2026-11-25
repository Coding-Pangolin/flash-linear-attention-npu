// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"
#include "chunk_kda_cp_fwd_tiling.h"
#include "chunk_kda_cp_fwd_tiling_key.h"
#include "device/chunk32_summary.h"

namespace {
using namespace AscendC;
using KcpWorkspace::kDim;
using KcpWorkspace::kRelayBatchBytes;
using KcpWorkspace::kRelayBatchHeads;
using KcpWorkspace::kStateBytes;
using KcpWorkspace::kSummaryBytes;
constexpr uint32_t kControlGroups = 1;
constexpr uint32_t kHalf = kDim / 2;

__aicore__ inline GM_ADDR RelayReceive(GM_ADDR workspace, const ChunkKdaCpFwdTilingData& td,
                                     uint32_t wave)
{
    return workspace + (1 + (wave & 1U) * td.ranks) * kRelayBatchBytes;
}

// Reserve IDs by direction to keep Merge separate from the chunk32 event schedule.
constexpr uint16_t kMergeInput = 1;   // AIV -> AIC
constexpr uint16_t kMergeOutput = 10; // AIC -> AIV
static_assert(KcpSummary32::kScratchBytes <= KcpWorkspace::kWorkerBytes);
static_assert(KcpSummary32::kPayload * sizeof(float) == kSummaryBytes);

struct Tensors {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR aLog;
    GM_ADDR bias;
    GM_ADDR offsets;
    GM_ADDR output;
    GM_ADDR finalState;
    GM_ADDR workspace;
};

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
template <bool Cube>
class Worker {
public:
    __aicore__ inline Worker(const Tensors& io, const ChunkKdaCpFwdTilingData& plan, uint32_t owner, uint32_t lane)
        : io_(io), td_(plan), owner_(owner), lane_(lane),
          workerScratch_(plan.stateRelay ? nullptr : io.workspace + plan.workerOffset),
          state_(plan.stateRelay ? nullptr : workerScratch_ + owner * KcpWorkspace::kWorkerBytes),
          offsets_(reinterpret_cast<__gm__ const int32_t*>(io.offsets))
    {}

    __aicore__ inline void Run()
    {
        if (td_.stateRelay != 0) {
            RelayForward();
            return;
        }
        const uint32_t first = td_.rounds == 2 ? (td_.heads + 1) / 2 : td_.heads;
        for (uint32_t batch = 0; batch < td_.rounds; ++batch) {
            Outbound(batch == 0 ? 0 : first, batch == 0 ? first : td_.heads);
            Barrier();
        }
        // For two rounds, consume the first half before the independent
        // tasks. The final barrier still joins the last CCU wait.
        const uint32_t independent = td_.firstSource < 0 ? 0 : td_.heads;
        for (uint32_t phase = 0; phase < 3; ++phase) {
            bool firstPairReady = false;
            if (phase == 2 && td_.rounds != 0) {
#if defined(__DAV_C310_VEC__)
                const uint32_t pendingHead = (td_.rounds == 2 ? first : 0) + owner_;
                if (td_.firstSource >= 0 && td_.segments == 1 && pendingHead < td_.heads) {
                    BindForward(0, pendingHead);
                    forward_.PrepareBoundTaskFirstPair();
                    firstPairReady = true;
                }
#endif
                Barrier();
            }
            uint32_t begin = 0;
            uint32_t end = 0;
            const bool merge = phase != 1;
            if (phase == 1) {
                begin = independent;
                end = (td_.segments - static_cast<uint32_t>(CanReuseTailForward())) * td_.heads;
            } else if (td_.firstSource >= 0) {
                if (phase == 0 && td_.rounds == 2) {
                    end = first;
                } else if (phase == 2 && td_.rounds != 0) {
                    begin = td_.rounds == 2 ? first : 0;
                    end = td_.heads;
                }
            }
            for (uint32_t task = begin + owner_; task < end; task += td_.workers) {
                const uint32_t head = task % td_.heads;
                if (merge) {
                    Merge(head);
                }
                Forward(task / td_.heads, head, merge ? state_ : nullptr, firstPairReady);
                firstPairReady = false;
            }
        }
    }

private:
    __aicore__ inline void RelayForward()
    {
        if constexpr (!Cube) {
            if (td_.rank != 0 && owner_ < kRelayBatchHeads) {
                // Initialize send slots before the first collective on an idle rank.
                LocalTensor<float> zero(TPosition::VECCALC, 0, kHalf * kDim);
                GlobalTensor<float> send;
                send.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    io_.workspace + owner_ * kStateBytes) + lane_ * kHalf * kDim);
                Duplicate(zero, 0.0f, kHalf * kDim);
                PipeBarrier<PIPE_ALL>();
                DataCopy(send, zero, kHalf * kDim);
                PipeBarrier<PIPE_ALL>();
            }
        }
        for (uint32_t wave = 0; wave <= td_.ranks; ++wave) {
            // The last rank has no outgoing H. Let CCU start this wave first.
            const bool computeDuringGather = td_.rank + 1 == td_.ranks && wave == td_.rank;
            if (computeDuringGather) {
                Barrier();
            }
            if (owner_ < kRelayBatchHeads && wave >= td_.rank && wave - td_.rank < td_.rounds) {
                const uint32_t head = (wave - td_.rank) * kRelayBatchHeads + owner_;
                GM_ADDR initial = nullptr;
                if (td_.rank > 0) {
                    initial = RelayReceive(io_.workspace, td_, wave - 1) +
                        (td_.rank - 1) * kRelayBatchBytes + owner_ * kStateBytes;
                }
                GM_ADDR outgoing = td_.rank + 1 < td_.ranks ?
                    io_.workspace + owner_ * kStateBytes : nullptr;
                Forward(0, head, initial, false, outgoing, true);
            }
            if (!computeDuringGather) {
                Barrier();
            }
            if (wave < td_.ranks) {
                Barrier();
            }
        }
    }

    __aicore__ inline void Barrier()
    {
        if constexpr (!Cube) {
            SyncAll<true>();
        }
    }

    __aicore__ inline bool CanReuseTailForward() const
    {
        // An incoming prefix belongs to the first segment. If the tail is a
        // different segment, it still starts locally and only needs to send E.
        return td_.sendTail && (td_.firstSource < 0 || td_.segments > 1);
    }

    __aicore__ inline void Outbound(uint32_t first, uint32_t end)
    {
        if (first + owner_ >= end) {
            return;
        }
        if (CanReuseTailForward()) {
            // Merge starts with E from the sequence's first participating rank.
            // Its local forward already produces that zero-initialized end state.
            for (uint32_t head = first + owner_; head < end; head += td_.workers) {
                Forward(td_.segments - 1, head, nullptr, false, io_.workspace + head * kSummaryBytes);
            }
            return;
        }
        if (td_.sendTail) {
            const uint32_t tail = (td_.segments - 1) * td_.heads;
            const KcpSummary32::Schedule schedule{
                td_.tokens, td_.heads, tail + end, td_.workers, io_.offsets,
                tail + first, tail, kControlGroups, KcpWorkspace::kWorkerBytes};
            // The rolling FP32 E/M buffer is already the send slot. The
            // outbound barrier releases it only after all last stores drain.
            KcpSummary32::Run<int32_t, true>(
                {io_.k, io_.v, io_.gate, io_.beta, io_.aLog, io_.bias, io_.workspace, workerScratch_, nullptr},
                schedule);
            return;
        }
        if constexpr (!Cube) {
            LocalTensor<float> zero(TPosition::VECCALC, 0, kHalf * 2 * kDim);
            Duplicate(zero, 0.0f, kHalf * 2 * kDim);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t head = first + owner_; head < end; head += td_.workers) {
                GlobalTensor<float> send;
                send.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(io_.workspace + head * kSummaryBytes));
                DataCopy(send[lane_ * kHalf * 2 * kDim], zero, kHalf * 2 * kDim);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void Merge(uint32_t head)
    {
        const uint32_t src0 = static_cast<uint32_t>(td_.firstSource);
        GM_ADDR recv = io_.workspace + (static_cast<uint64_t>(td_.heads) * (1 + src0) + head) * kSummaryBytes;
        GM_ADDR delta = state_ + kStateBytes;
        if constexpr (Cube) {
            TPipe pipe;
            KcpState::ChunkGemmTiles<64, 128, float> gemm;
            gemm.Init(pipe);
            for (uint32_t src = src0 + 1; src < td_.rank; ++src) {
                recv += td_.heads * kSummaryBytes;
                CrossCoreWaitFlag<4, PIPE_MTE2>(kMergeInput);
                CrossCoreWaitFlag<4, PIPE_MTE2>(kMergeInput + 16);
                gemm.template Run<kDim, kDim, kDim, float>(recv + kDim * sizeof(float), state_, delta, 2 * kDim);
                CrossCoreSetFlag<4, PIPE_FIX>(kMergeOutput);
                CrossCoreSetFlag<4, PIPE_FIX>(kMergeOutput + 16);
            }
        } else {
            Rows(recv, nullptr);
            for (uint32_t src = src0 + 1; src < td_.rank; ++src) {
                recv += td_.heads * kSummaryBytes;
                CrossCoreSetFlag<4, PIPE_MTE3>(kMergeInput);
                CrossCoreWaitFlag<4, PIPE_MTE2>(kMergeOutput);
                Rows(recv, delta);
            }
        }
    }

    __aicore__ inline void Forward(uint32_t seq, uint32_t head, GM_ADDR initial, bool firstPairReady, GM_ADDR outgoing = nullptr, bool compactOutgoing = false)
    {
        const uint32_t rows = offsets_[seq + 1] - offsets_[seq];
        const uint32_t windows = (rows + tokens_per_group - 1) / tokens_per_group;
#if defined(__DAV_C310_CUBE__)
        forward_.Init();
        forward_.ProcessTask(windows);
#elif defined(__DAV_C310_VEC__)
        if (!firstPairReady) {
            BindForward(seq, head);
        }
        if (initial != nullptr) {
            forward_.has_istate = 1;
            forward_.istate_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial) + lane_ * kHalf * kDim);
        }
        forward_.ProcessBoundTask(windows, firstPairReady, outgoing, compactOutgoing);
        PipeBarrier<PIPE_ALL>();
#endif
    }

#if defined(__DAV_C310_VEC__)
    __aicore__ inline void BindForward(uint32_t seq, uint32_t head)
    {
        const uint32_t rows = offsets_[seq + 1] - offsets_[seq];
        const uint32_t windows = (rows + tokens_per_group - 1) / tokens_per_group;
        const uint32_t task = seq * td_.heads + head;
        union { float value; uint32_t bits; } scale{td_.scale}, gate{-5.0f}, epsilon{1.0e-6f};
        forward_.Init(windows, scale.bits >> 16, gate.bits, io_.aLog, io_.gate, io_.q, io_.k, io_.v, io_.beta,
                      io_.bias, io_.output, io_.output, nullptr, io_.finalState, 0, td_.saveFinal, td_.tokens,
                      1, td_.heads * td_.segments, td_.heads, td_.workers, 2,
                      reinterpret_cast<uint64_t>(io_.offsets), epsilon.bits, 3);
        forward_.template BindTaskVarlen<int32_t, true>(task);
    }
#endif

    __aicore__ inline void Rows(GM_ADDR src, GM_ADDR add)
    {
        GlobalTensor<float> input, dest, delta;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
        dest.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state_));
        constexpr uint32_t tileRows = 16;
        constexpr uint32_t tileElems = tileRows * kDim;
        constexpr uint16_t rowBlocks = kDim * sizeof(float) / 32;
        static_assert(kHalf % tileRows == 0 && 2 * tileElems * sizeof(float) <= 16 * 1024);
        // The first prepared Q/K pair may be live above this UB range.
        LocalTensor<float> local(TPosition::VECCALC, 0, 2 * tileElems);
        auto update = local[tileElems];
        const DataCopyParams gather{tileRows, rowBlocks, rowBlocks, 0};
        if (add != nullptr) {
            delta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(add));
        }
        for (uint32_t row = lane_ * kHalf; row < (lane_ + 1) * kHalf; row += tileRows) {
            const uint32_t offset = row * kDim;
            DataCopy(local, input[2 * offset], gather);
            if (add != nullptr) {
                DataCopy(update, delta[offset], tileElems);
                PipeBarrier<PIPE_ALL>();
                Add(local, local, update, tileElems);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(dest[offset], local, tileElems);
            PipeBarrier<PIPE_ALL>();
        }
    }

    const Tensors& io_;
    const ChunkKdaCpFwdTilingData& td_;
    uint32_t owner_;
    uint32_t lane_;
    GM_ADDR workerScratch_;
    GM_ADDR state_;
    __gm__ const int32_t* offsets_;
    // The original pipeline retains its CrossCore flag phase across tasks.
#if defined(__DAV_C310_CUBE__)
    Kernel<bfloat16_t> forward_;
#else
    Kernel<bfloat16_t, true, true> forward_;
#endif
};
#endif

__aicore__ inline void Communicate(const ChunkKdaCpFwdTilingData& td, GM_ADDR workspace)
{
    constexpr uint64_t elementsPerHead = kSummaryBytes / sizeof(float);
    const uint64_t count = td.heads * elementsPerHead;
    const uint64_t first = td.rounds == 2 ? ((td.heads + 1) / 2) * elementsPerHead : count;
    GM_ADDR gather = workspace + td.heads * kSummaryBytes;
    Hccl<HCCL_SERVER_TYPE_CCU> hccl;
    hccl.InitV2(GetHcclContext<0>(), &td.mc2InitTiling);
    hccl.SetCcTilingV2(offsetof(ChunkKdaCpFwdTilingData, mc2CcTiling));
    // The MC2 executor still expects its service lifecycle to close when
    // boundary-aligned packed sequences schedule no data collective.
    if (td.rounds == 0) {
        hccl.Finalize();
        return;
    }
    if (td.stateRelay != 0) {
        for (uint32_t wave = 0; wave <= td.ranks; ++wave) {
            SyncAll<true>();
            if (wave < td.ranks) {
                auto handle = hccl.AllGather<true>(workspace, RelayReceive(workspace, td, wave),
                    kRelayBatchBytes / sizeof(float), HCCL_DATA_TYPE_FP32, kRelayBatchBytes / sizeof(float));
                hccl.Wait(handle);
                SyncAll<true>();
            }
        }
        hccl.Finalize();
        return;
    }
    SyncAll<true>();
    auto handle = hccl.AllGather<true>(workspace, gather, first, HCCL_DATA_TYPE_FP32, count);
    hccl.Wait(handle);
    SyncAll<true>();
    if (td.rounds == 2) {
        const uint64_t offset = first * sizeof(float);
        auto next = hccl.AllGather<true>(workspace + offset, gather + offset, count - first, HCCL_DATA_TYPE_FP32, count);
        hccl.Wait(next);
        SyncAll<true>();
    }
    hccl.Finalize();
}
} // namespace

template <int D_T>
__global__ __aicore__ void chunk_kda_cp_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gate, GM_ADDR beta, GM_ADDR aLog, GM_ADDR bias,
    GM_ADDR offsets, GM_ADDR output, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(D_T == 10, "KCP v1 uses BF16 q/k/v");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkKdaCpFwdTilingData);
    GET_TILING_DATA(td, tiling);
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    constexpr uint32_t control = kControlGroups;
    const Tensors io{q, k, v, gate, beta, aLog, bias, offsets, output, finalState, workspace};
    if ASCEND_IS_AIV {
        const uint32_t group = GetBlockIdx() / GetSubBlockNum();
        if (group < control) {
            Communicate(td, workspace);
        } else {
            Worker<false> compute(io, td, group - control, GetSubBlockIdx());
            compute.Run();
        }
    }
    if ASCEND_IS_AIC {
        const uint32_t group = GetBlockIdx();
        if (group >= control) {
            Worker<true> compute(io, td, group - control, 0);
            compute.Run();
        }
    }
#endif
}
