/**
 * bench_fp32_mmad — CATLASS fp32 (float x float -> float) matmul 微基准
 * 改写自 catlass examples/68_ascend950_multi_core_splitk_matmul。
 * 量 M_c @ m 形状的 fp32 matmul 在 950PR 上一步耗时, 以及 HF32 开/关的差异。
 * 用法: ./99_bench_fp32_mmad [M] [N] [K] [device] [iters]
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <chrono>
#include <cstdio>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/multi_core_splitk_matmul_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

template <bool useHF32>
static double BenchOne(const Options& options, uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum,
                       uint8_t* deviceA, uint8_t* deviceB, uint8_t* deviceC,
                       uint8_t* deviceBias, aclrtStream stream, int iters)
{
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementBias = void;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;
    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;
    using MatmulKernel = Gemm::Kernel::MultiCoreSplitkMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    typename MatmulKernel::Arguments arguments{options.problemShape, deviceA, layoutA, deviceB, layoutB,
                                      deviceC, layoutC, aicCoreNum, deviceBias};
    MatmulAdapter matmul_op;
    matmul_op.CanImplement(arguments);

    uint8_t* deviceWorkspace{nullptr};
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);

    matmul_op(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        matmul_op(stream, aicCoreNum);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::steady_clock::now();
    double per = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)iters;

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    return per;
}

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    const int iters = 200;

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    size_t lenA = tagA.Capacity(), lenB = tagB.Capacity(), lenC = tagC.Capacity();
    size_t sizeA = lenA * sizeof(ElementA), sizeB = lenB * sizeof(ElementB), sizeC = lenC * sizeof(ElementC);

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    golden::FillRandomData<ElementA>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<ElementB>(hostB, -5.0f, 5.0f);

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    double perNative = BenchOne<false>(options, m, n, k, aicCoreNum, deviceA, deviceB, deviceC, nullptr, stream, iters);
    double perHF32 = BenchOne<true>(options, m, n, k, aicCoreNum, deviceA, deviceB, deviceC, nullptr, stream, iters);

    double macs = (double)m * n * k;
    printf("shape m=%u n=%u k=%u  aicCoreNum=%u  iters=%d\n", m, n, k, aicCoreNum, iters);
    printf("  FP32(native, HF32=off): %10.3f us/call  %8.2f GMAC/s  %7.1f GFLOPS\n",
           perNative, macs / perNative / 1e3, 2 * macs / perNative / 1e6);
    printf("  FP32(HF32=on)         : %10.3f us/call  %8.2f GMAC/s  %7.1f GFLOPS\n",
           perHF32, macs / perHF32 / 1e3, 2 * macs / perHF32 / 1e6);
    printf("  HF32 speedup          : %10.2fx\n", perNative / perHF32);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}