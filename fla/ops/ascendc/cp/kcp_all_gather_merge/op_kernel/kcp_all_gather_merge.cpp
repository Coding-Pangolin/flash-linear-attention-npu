// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
// HCCL lifecycle follows Huawei's AllGatherAdd example (CANN OSL v2).
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "adv_api/hccl/hccl.h"
#include "kcp_all_gather_merge_tiling.h"
#include "kcp_all_gather_merge_tiling_key.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#include "kcp_merge_reference/arch35/merge_fwd_bwd_kernel_cube.h"
#include "kcp_merge_reference/arch35/merge_fwd_bwd_kernel_vector.h"
#else
#include "kcp_merge_reference/merge_fwd_bwd_kernel_cube.h"
#include "kcp_merge_reference/merge_fwd_bwd_kernel_vector.h"
#endif

using namespace AscendC;

#if defined(FLA_KCP_A5_PROGRESS_PROBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
// The opt-in diagnostic reserves the first 16 FP32 words of rank 0's H0
// (normally zero). The two aligned rows below have distinct writers.
__aicore__ inline void WriteProgressProbe(
    GM_ADDR h0, int32_t row, int64_t begin, int64_t end, int64_t aux0, int64_t aux1 = 0)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> local;
    pipe.InitBuffer(local, 32);
    auto counters = local.Get<int64_t>();
    counters.SetValue(0, begin);
    counters.SetValue(1, end);
    counters.SetValue(2, aux0);
    counters.SetValue(3, aux1);
    GlobalTensor<int64_t> record;
    record.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(h0));
    PipeBarrier<PIPE_ALL>();
    DataCopy(record[row * 4], counters, 4);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunIndependentProbeWork(GM_ADDR h0, bool record)
{
    int64_t begin, end, result;
    {
        TPipe pipe;
        TBuf<TPosition::VECCALC> work;
        pipe.InitBuffer(work, 128 * sizeof(float));
        auto values = work.Get<float>();
        begin = GetSystemCycle();
        Duplicate(values, 1.0f, 128);
        for (int i = 0; i < 4096; ++i) {
            PipeBarrier<PIPE_V>();
            Muls(values, values, 1.0001f, 128);
        }
        PipeBarrier<PIPE_ALL>();
        end = GetSystemCycle();
        result = static_cast<int64_t>(values.GetValue(0) * 1000000.0f);
    }
    if (record) {
        WriteProgressProbe(h0, 1, begin, end, result);
    }
}

#if FLA_KCP_A5_PROGRESS_PROBE == 3
// CP2 has one upstream transform. Its exclusive prefix is E from physical
// gather slot 0. Only the CCU leader reads it after its completion wait.
__aicore__ inline void CopyCp2Prefix(GM_ADDR gathered, GM_ADDR h0, uint32_t firstHead, uint32_t lastHead)
{
    constexpr uint32_t kDim = 128;
    constexpr uint32_t kRows = 32;
    constexpr uint16_t kBlocks = kDim * sizeof(float) / 32;
    TPipe pipe;
    TBuf<TPosition::VECCALC> local;
    pipe.InitBuffer(local, kRows * kDim * sizeof(float));
    auto values = local.Get<float>();
    GlobalTensor<float> recv, out;
    recv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gathered));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h0));
    for (uint32_t head = firstHead; head < lastHead; ++head) {
        for (uint32_t row = 0; row < kDim; row += kRows) {
            const uint32_t offset = (head * kDim + row) * kDim;
            DataCopy(values, recv[offset * 2], DataCopyParams{kRows, kBlocks, kBlocks, 0});
            PipeBarrier<PIPE_ALL>();
            DataCopy(out[offset], values, kRows * kDim);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
#endif
#endif

template <int D_T>
__global__ __aicore__ void kcp_all_gather_merge(
    GM_ADDR summary, GM_ADDR hIn, GM_ADDR h0, GM_ADDR gathered,
    GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(D_T == 30, "KCP transport requires FP32");
    (void)hIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(KcpAllGatherMergeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    auto &m = tilingData.merge;
    GM_ADDR scratch = workspace;
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
    scratch = GetUserWorkspace(workspace);
    if (scratch == nullptr) scratch = workspace + m.sysWorkspaceSize;
#endif
    if ASCEND_IS_AIV {
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if (GetBlockIdx() / GetSubBlockNum() != 0) {
            RunIndependentProbeWork(h0, m.rank == 0 && GetSubBlockIdx() == 0);
            SyncAll<true>();
            return;
        }
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        Hccl<HCCL_SERVER_TYPE_CCU> hccl;
#else
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
#endif
        hccl.InitV2(GetHcclContext<0>(), &tilingData.mc2InitTiling);
        hccl.SetCcTilingV2(offsetof(KcpAllGatherMergeTilingData, mc2CcTiling));
        constexpr uint64_t summaryPerHead = 128ULL * 256ULL;
        const uint64_t count = static_cast<uint64_t>(m.Hv) * summaryPerHead;
        uint64_t first = count;
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && FLA_KCP_A5_PROGRESS_PROBE >= 2 && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        first = ((static_cast<uint64_t>(m.Hv) + 1) / 2) * summaryPerHead;
#endif
        auto handle = hccl.AllGather<true>(summary, gathered, first, HCCL_DATA_TYPE_FP32, count);
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        const int64_t submitted = GetSystemCycle();
#endif
        hccl.Wait(handle);
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        const int64_t ready = GetSystemCycle();
        int64_t secondSubmit = 0, secondReady = 0;
#if FLA_KCP_A5_PROGRESS_PROBE == 3
        const bool consume = m.rank == 1 && GetSubBlockIdx() == 0;
        const uint32_t firstHeads = first / summaryPerHead;
        int64_t mergeBegin = 0, mergeEnd = 0;
#endif
#if FLA_KCP_A5_PROGRESS_PROBE >= 2
        if (first < count) {
            const uint64_t byteOffset = first * sizeof(float);
            auto next = hccl.AllGather<true>(summary + byteOffset, gathered + byteOffset,
                                            count - first, HCCL_DATA_TYPE_FP32, count);
            secondSubmit = GetSystemCycle();
#if FLA_KCP_A5_PROGRESS_PROBE == 3
            if (consume) {
                mergeBegin = GetSystemCycle();
                CopyCp2Prefix(gathered, h0, 0, firstHeads);
                mergeEnd = GetSystemCycle();
            }
#endif
            hccl.Wait(next);
            secondReady = GetSystemCycle();
        }
#endif
#if FLA_KCP_A5_PROGRESS_PROBE == 3
        if (consume) {
            CopyCp2Prefix(gathered, h0, firstHeads, m.Hv);
            WriteProgressProbe(h0, 0, submitted, ready, secondSubmit, secondReady);
            WriteProgressProbe(h0, 1, mergeBegin, mergeEnd, 0);
        }
#else
        if (m.rank == 0 && GetSubBlockIdx() == 0) {
            WriteProgressProbe(h0, 0, submitted, ready, secondSubmit, secondReady);
        }
#endif
#endif
        SyncAll<true>();
        hccl.Finalize();
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && FLA_KCP_A5_PROGRESS_PROBE == 3 && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        return;
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        CrossCoreSetFlag<0x4, PIPE_MTE3>(8);
#else
        CrossCoreSetFlag<0x2, PIPE_MTE3>(8);
#endif
        if (m.rank > 0) {
            TPipe pipe;
            MergeFwBwd::MergeFwdBwdKernelVectorProcess<float> vector(gathered, scratch, h0, scratch, &m);
            vector.Init(&pipe);
            vector.Process();
        }
    }
    if ASCEND_IS_AIC {
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && FLA_KCP_A5_PROGRESS_PROBE == 3 && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        return;
#endif
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if (GetBlockIdx() != 0) {
            return;
        }
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(8);
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(24);
#else
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(8);
#endif
        if (m.rank > 0) {
            MergeFwBwd::MergeFwdBwdKernelCubeProcess<float> cube(gathered, scratch, h0, scratch);
            cube.Init(&m);
            cube.Process();
        }
    }
}
