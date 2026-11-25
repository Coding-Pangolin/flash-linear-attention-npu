#pragma once

#include "chunk_gemm.h"
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#if __has_include("../../chunk_kda_fwd_fused/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h")
#include "../../chunk_kda_fwd_fused/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h"
#elif __has_include("../../../chunk_kda_fwd_fused/op_kernel/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h")
#include "../../../chunk_kda_fwd_fused/op_kernel/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h"
#else
#include "../../../../kda/chunk_kda_fwd_fused/op_kernel/arch35/chunk_kda_fwd_fused/chunk_kda_fwd_32.h"
#endif
#endif

namespace KcpSummary32 {
constexpr uint32_t kRows = 32;
constexpr uint32_t kDim = 128;
constexpr uint32_t kWidth = 2 * kDim;
constexpr uint32_t kRowsPerPair = 2 * kRows;
constexpr uint32_t kWindow = 2 * kRowsPerPair;
constexpr uint32_t kPayload = kDim * kWidth;
constexpr uint32_t kCorrectionBytes = kRows * kWidth * sizeof(bfloat16_t);
constexpr uint32_t kOperandBanks = 2;
constexpr uint32_t kStateL1Bytes = kPayload * sizeof(bfloat16_t);
constexpr uint32_t kDecayBytes = kDim / 2 * sizeof(float);
constexpr uint32_t kDecaySlotStride = kOperandBanks * 2 * kDecayBytes;
// Two banks of decay tiles, one tile per producing AIV in each bank.
constexpr uint32_t kScratchBytes = kDecaySlotStride;
static_assert(kScratchBytes % kDecaySlotStride == 0);
// Summary-only L1 residency, above the K/K score slots and the generic GEMM buffers.
constexpr uint32_t kStateL1 = 256 * 1024;
constexpr uint32_t kWL1 = 336 * 1024;
static_assert(kStateL1 + kStateL1Bytes <= kWL1);
static_assert(kWL1 + kRows * kDim * sizeof(bfloat16_t) <= 512 * 1024);
// Two pair banks, one 32x128 NZ operand per producer AIV.
constexpr uint32_t kNextWL1 = kWL1 + kRows * kDim * sizeof(bfloat16_t);
constexpr uint32_t kKeyResL1 = 352 * 1024;
constexpr uint32_t kKeyResBytes = kRows * kDim * sizeof(bfloat16_t);
static_assert(kNextWL1 + kRows * kDim * sizeof(bfloat16_t) <= kKeyResL1);
static_assert(kKeyResL1 + kOperandBanks * 2 * kKeyResBytes <= 512 * 1024);
__aicore__ inline uint32_t KeyResL1(uint32_t pair, uint32_t chunk)
{
    return kKeyResL1 + ((pair % kOperandBanks) * 2 + chunk) * kKeyResBytes;
}
constexpr uint32_t kRhsL1 = 384 * 1024;
constexpr uint32_t kRhsL1Bytes = kRows * kWidth * sizeof(bfloat16_t);
constexpr uint32_t kRhsL1End = kRhsL1 + kOperandBanks * 2 * kRhsL1Bytes;
static_assert(kKeyResL1 + kOperandBanks * 2 * kKeyResBytes <= kRhsL1);
static_assert(kRhsL1End <= 512 * 1024);
__aicore__ inline uint32_t RhsL1(uint32_t pair, uint32_t chunk)
{
    return kRhsL1 + ((pair % kOperandBanks) * 2 + chunk) * kRhsL1Bytes;
}
// The current pair drains before the next inverse is produced.
constexpr uint32_t kInverseL1 = kRhsL1End;
constexpr uint32_t kInverseL1Bytes = kRows * kRows * sizeof(bfloat16_t);
static_assert(kInverseL1 + 2 * kInverseL1Bytes <= 512 * 1024);
__aicore__ inline uint32_t InverseL1(uint32_t chunk)
{
    return kInverseL1 + chunk * kInverseL1Bytes;
}
// The summary has drained its RHS before this range holds FP32 M.
constexpr uint32_t kAffineMatrixL1 = kRhsL1;
static_assert(kAffineMatrixL1 + kDim * kDim * sizeof(float) <= kRhsL1End);
constexpr uint32_t kStateUb = 120 * 1024;
constexpr uint32_t kCorrectionL1 = kInverseL1 + 2 * kInverseL1Bytes;
static_assert(kCorrectionL1 + kCorrectionBytes <= 512 * 1024);
constexpr uint32_t kExchangeUb = 84 * 1024;
// U follows the FP32 projections in each AIV's UB.
constexpr uint32_t kUUb = kExchangeUb + (kRows / 2) * kWidth * sizeof(float);
constexpr uint32_t kNextUUb = 116 * 1024;
// Reuse the raw input area after K factors have been packed and their loads drained.
constexpr uint32_t kDeltaUb = 184 * 1024;
static_assert(kDeltaUb + kPayload * sizeof(float) / 2 <= 256 * 1024);
constexpr uint32_t kTracePairs = 8;
constexpr uint32_t kTraceFields = 17;
// AIC/AIV scalar stores must not share a GM cache line.
constexpr uint32_t kTraceStride = 16;
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
static_assert(kRows == chunk_size && kWindow == tokens_per_group && kDim == key_dim);
#endif

struct Inputs {
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR alpha;  // exp(A_log), as supplied to the original chunk32 kernel.
    GM_ADDR bias;
    GM_ADDR payload;
    GM_ADDR scratch;
    GM_ADDR trace;
};

struct Schedule {
    uint32_t tokens;
    uint32_t heads;
    uint32_t tasks;
    uint32_t groups;
    GM_ADDR offsets;
    uint32_t firstTask = 0;
    uint32_t payloadBase = 0;
    uint32_t controlGroups = 0;
    uint32_t scratchStride = kScratchBytes;
    // Default callers distribute the contiguous task range. The head relay can bind one task.
    uint32_t boundTask = static_cast<uint32_t>(-1);
};

__aicore__ inline GM_ADDR DecaySlot(GM_ADDR scratch, uint32_t pair, uint32_t lane)
{
    return scratch + ((pair % kOperandBanks) * 2 + lane) * kDecayBytes;
}

enum Flag : uint16_t {
    kOperandsReady,
    kScoresReady,
    kInverseReady,
    kProductsReady,
    kCorrectionReady,
    kResultReady,
    kStateStored,
    kDeltaSlotFree = 5,  // AIV -> AIC; kResultReady uses the reverse direction.
};

template<Flag Id>
__aicore__ inline void WaitLanes()
{
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(Id);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(Id + 16);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<Flag Id>
__aicore__ inline void SendLanes()
{
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(Id);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(Id + 16);
}

template<Flag Id>
__aicore__ inline void WaitCube()
{
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(Id);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<Flag Id>
__aicore__ inline void SendCube()
{
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(Id);
}

__aicore__ inline void Mark(const Inputs& input, uint32_t task, uint32_t pair, uint32_t field)
{
    if (input.trace != nullptr && task == 0 && pair < kTracePairs) {
        auto* times = reinterpret_cast<__gm__ int64_t*>(input.trace);
        times[(pair * kTraceFields + field) * kTraceStride] = AscendC::GetSystemCycle();
    }
}

__aicore__ inline uint32_t PairChunks(uint32_t totalChunks, uint32_t pair)
{
    return totalChunks - pair * 2 == 1 ? 1 : 2;
}

#if defined(__DAV_C310_CUBE__)
__aicore__ inline void FormWu(
    Kernel<bfloat16_t>& original, KcpState::ChunkGemmTiles<kDim, kWidth, bfloat16_t>& gemm,
    uint32_t inverseOffset, uint32_t rhsOffset, uint32_t wOffset, uint32_t uOffset)
{
    gemm.ComputeWithRhsInL1<kRows, kWidth, kRows, true>(nullptr, inverseOffset, rhsOffset);
    const auto wu = gemm.ResultL0C();
    const auto u = wu[kRows * kDim];
    original.template copy_l0c_to_l1_bf16<kRows, kDim>(L1b<bfloat16_t>(wOffset), wu);
    original.template copy_l0c_to_ub_bf16<true, 0, kRows, kDim>(Ub<bfloat16_t>(uOffset), u);
    original.template copy_l0c_to_ub_bf16<true, 1, kRows, kDim>(
        Ub<bfloat16_t>(uOffset), u[(kRows / 2) * 16]);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<typename Offset = int64_t, bool LogAlpha = false>
__aicore__ inline void Run(const Inputs& input, const Schedule& schedule)
{
    const uint32_t group = AscendC::GetBlockIdx() - schedule.controlGroups;
    auto* offsets = reinterpret_cast<__gm__ const Offset*>(schedule.offsets);
    const bool bound = schedule.boundTask < schedule.tasks;
    Kernel<bfloat16_t> original;
    original.Init();
    AscendC::TPipe pipe;
    KcpState::ChunkGemmTiles<128, 256, bfloat16_t> gemm;
    gemm.Init(pipe);

    const uint32_t firstTask = bound ? schedule.boundTask : schedule.firstTask + group;
    const uint32_t taskEnd = bound ? schedule.boundTask + 1 : schedule.tasks;
    for (uint32_t task = firstTask; task < taskEnd; task += schedule.groups) {
        const uint32_t seq = task / schedule.heads;
        const uint32_t rows = offsets[seq + 1] - offsets[seq];
        const uint32_t chunks = (rows + kRows - 1) / kRows;
        const uint32_t pairs = (rows + kRowsPerPair - 1) / kRowsPerPair;
        GM_ADDR state = input.payload + static_cast<uint64_t>(task - schedule.payloadBase) * kPayload * sizeof(float);
        // A window consists of pairs (c0,c1) and (c2,c3). AIC issues the original
        // four rectangular K/K MMADs per pair and advances only valid states.
        for (uint32_t pair = 0; pair < pairs; ++pair) {
            WaitLanes<kOperandsReady>();
            original.copy_l1_to_l0ab_kplus_kminus(
                0, original.kplus_kminus_a_slot, original.kplus_kminus_b_slot, original.kd_auto_unroll_id_1_slot, original.kd2_auto_unroll_id_1_slot, original.ki_auto_unroll_id_1_slot);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            original.matmul_kplus_kminus(
                0, original.kplus_kminus_c_slot, original.kplus_kminus_a_slot, original.kplus_kminus_b_slot);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
            const auto score = L0c<float>(original.addrL0C[original.kplus_kminus_c_slot] * l0_addr_unit);
            original.template copy_l0c_to_ub_f32<false, 2 * kRows, kRows>(Ub<float>(kExchangeUb), score);
            SendLanes<kScoresReady>();
            WaitLanes<kInverseReady>();

            Mark(input, task, pair, 6);  // First factor/application GEMMs begin.
            const uint32_t count = PairChunks(chunks, pair);
            FormWu(original, gemm, InverseL1(0), RhsL1(pair, 0), kWL1, kUUb);
            for (uint32_t chunk = 0; chunk < count; ++chunk) {
                gemm.ProjectFromL1<kRows, kWidth, kDim>(
                    chunk == 0 ? kWL1 : kNextWL1, kStateL1, kExchangeUb);
                if (chunk == 0) {
                    Mark(input, task, pair, 7);
                }
                SendLanes<kProductsReady>();
                if (chunk == 0 && count == 2) {
                    FormWu(original, gemm, InverseL1(1), RhsL1(pair, 1), kNextWL1, kNextUUb);
                }
                WaitLanes<kCorrectionReady>();
                if (chunk == 0) Mark(input, task, pair, 15);
                // Keep the full Delta MMAD and reuse one UB exchange area for E and M.
                gemm.ComputeFromTransposedL1Both<kDim, kWidth, kRows>(KeyResL1(pair, chunk), kCorrectionL1);
                if (chunk == 0 && pair + 1 < pairs) {
                    // Both AIVs must finish packing the next pair before Fixpipe reuses UB.
                    WaitLanes<kDeltaSlotFree>();
                }
                if (chunk == 0) Mark(input, task, pair, 16);  // L0C ready to publish.
                gemm.PublishCols<kDim, kWidth>(0, kDeltaUb);
                SendLanes<kResultReady>();
                WaitLanes<kStateStored>();
            }
            Mark(input, task, pair, 8);
        }
    }
}
#elif defined(__DAV_C310_VEC__)
using Original = Kernel<bfloat16_t, true, true>;

// The reused raw loader uses UB [0,16 KiB) and addresses at and above 184 KiB.
// The remaining regions below are reserved solely by this K-only path.
constexpr uint32_t KiB = 1024;
constexpr uint32_t kGate = 16 * KiB;
constexpr uint32_t kNormAux = 32 * KiB;
constexpr uint32_t kPlus = 48 * KiB;
constexpr uint32_t kMinus = 57 * KiB;
constexpr uint32_t kRes = 66 * KiB;
constexpr uint32_t kPlusB = 75 * KiB;
constexpr uint32_t kBeta = 80 * KiB;
constexpr uint32_t kDecay = 81 * KiB;
constexpr uint32_t kRhsTile = kExchangeUb;
constexpr uint32_t kTriangular = 89 * KiB;
constexpr uint32_t kInvTmp = 95 * KiB;
constexpr uint32_t kInverse = 104 * KiB;
constexpr uint32_t kResidentUb = kStateUb;
static_assert(kNextUUb + (kRows / 2) * kDim * sizeof(bfloat16_t) <= kResidentUb);
constexpr uint32_t kLaneRows = kDim / 2;
constexpr uint32_t kStateRowsPerTile = 16;
static_assert(kResidentUb + kLaneRows * kWidth * sizeof(float) <= 184 * KiB);

__aicore__ inline __ubuf__ void* RawUb(uint32_t bytes)
{
    return reinterpret_cast<__ubuf__ void*>(Ub(bytes).GetPhyAddr());
}

template<typename T>
__aicore__ inline AscendC::LocalTensor<T> TileUb(uint32_t bytes, uint32_t elements)
{
    return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, bytes, elements);
}

template<uint32_t Bank>
__aicore__ inline void Prepare(Original& original)
{
    original.template vec_prepare_g3<Bank, true, true, true, true, false>(
        RawUb(kGate + Bank * 8 * KiB), RawUb(kDecay + Bank * 512),
        RawUb(kBeta + Bank * 256), nullptr, RawUb(kNormAux + Bank * 8 * KiB));
}

template<uint32_t Bank>
__aicore__ inline void BuildKFactors(Original& original)
{
    using namespace AscendC;
    original.template vec_prepare_qplus_kplus_kminus_kres<Bank, false>(
        nullptr, nullptr, RawUb(kPlus), RawUb(kPlusB), RawUb(kMinus), RawUb(kRes),
        RawUb(kGate + Bank * 8 * KiB), RawUb(kDecay + Bank * 512));
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CopyScoreFactorsToL1(Original& original)
{
    const uint32_t lane = original.subid;
    original.template ub_l1_src_strided_nz<kRows, kDim, 2>(
        Ub(kPlus), L1b(original.addrL1[original.kd_auto_unroll_id_1] * l1_addr_unit + lane * kRows * kDim * 2));
    original.template ub_l1_src_strided_nz<kRows / 2, kDim, 2>(
        Ub(kPlusB), L1b(original.addrL1[original.kd2_auto_unroll_id_1] * l1_addr_unit + lane * kRows * kDim));
    original.template ub_l1_src_strided_nz<kRows, kDim, 2>(
        Ub(kMinus), L1b(original.addrL1[original.ki_auto_unroll_id_1] * l1_addr_unit + lane * kRows * kDim * 2));
}

template<typename T, uint32_t Columns, uint32_t NzStride>
__aicore__ inline void UnpackNz(const AscendC::LocalTensor<T>& nd,
                               const AscendC::LocalTensor<T>& nz)
{
    using namespace AscendC;
    constexpr uint32_t l = 16;
    for (uint32_t col = 0; col < Columns / l; ++col) {
        DataCopy(nd[col * l], nz[col * NzStride * l],
                 DataCopyParams{static_cast<uint16_t>(kRows), 1, 0, static_cast<uint16_t>(Columns / l - 1)});
    }
}

__aicore__ inline void ExportOperands(Original& original, GM_ADDR decay, uint32_t pair,
                                      uint32_t rows, uint32_t bank)
{
    using namespace AscendC;
    const uint32_t lane = original.subid;
    const uint32_t start = pair * kRowsPerPair + lane * kRows;
    const uint32_t valid = start >= rows ? 0 : (rows - start < kRows ? rows - start : kRows);
    constexpr uint32_t ndElements = kRows * kDim;
    constexpr uint32_t bfBytes = ndElements * sizeof(bfloat16_t);
    constexpr uint32_t packOffset = kRhsTile + 2 * bfBytes;
    static_assert(packOffset + 2 * bfBytes <= kNextUUb);
    auto kNd = TileUb<bfloat16_t>(kRhsTile, ndElements);
    auto vNd = TileUb<bfloat16_t>(kRhsTile + bfBytes, ndElements);
    auto both = TileUb<bfloat16_t>(packOffset, kRows * kWidth);
    auto beta = TileUb<float>(kBeta + bank * 256, kRows);
    GlobalTensor<float> d;
    d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay));

    UnpackNz<bfloat16_t, kDim, 33>(kNd, TileUb<bfloat16_t>(kPlus, 33 * kDim));
    if (valid < kRows) {
        Duplicate(vNd.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), ndElements);
        PipeBarrier<PIPE_ALL>();
    }
    if (valid != 0) {
        DataCopy(vNd, original.v_gm[start * kDim], valid * kDim);
    }
    PipeBarrier<PIPE_ALL>();
    auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(both.GetPhyAddr());
    auto* k = reinterpret_cast<__ubuf__ bfloat16_t*>(kNd.GetPhyAddr());
    auto* v = reinterpret_cast<__ubuf__ bfloat16_t*>(vNd.GetPhyAddr());
    auto* b = reinterpret_cast<__ubuf__ float*>(beta.GetPhyAddr());
    __VEC_SCOPE__ {
        MaskReg all = CreateMask<bfloat16_t, MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(kRows); ++row) {
            RegTensor<float> scale;
            DataCopy<float, LoadDist::DIST_BRC_B32>(scale, b + row);
            for (uint16_t side = 0; side < 2; ++side) {
                RegTensor<bfloat16_t> x, lo, hi;
                RegTensor<float> even, odd;
                DataCopy(x, (side == 0 ? k : v) + row * kDim);
                Cast<float, bfloat16_t, bf2fZero>(even, x, all);
                Cast<float, bfloat16_t, bf2fOne>(odd, x, all);
                Mul(even, even, scale, all);
                Mul(odd, odd, scale, all);
                Cast<bfloat16_t, float, f2bfZero>(lo, even, all);
                Cast<bfloat16_t, float, f2bfOne>(hi, odd, all);
                Or(reinterpret_cast<RegTensor<uint16_t>&>(lo), reinterpret_cast<RegTensor<uint16_t>&>(lo),
                   reinterpret_cast<RegTensor<uint16_t>&>(hi), all);
                const uint16_t nzOffset = side * kRows * kDim + row * 16;
                DataCopy<bfloat16_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    out + nzOffset, lo, kRows, all);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    original.template ub_l1_single<kRows, kWidth, sizeof(bfloat16_t)>(
        Ub(packOffset), L1b(RhsL1(pair, lane)));

    original.template ub_l1_src_strided_nz<kRows, kDim, sizeof(bfloat16_t)>(
        Ub(kRes), L1b(KeyResL1(pair, lane)));
    // The producer's half is still available in its own UB.
    const uint32_t remoteHalf = (1 - lane) * kLaneRows * sizeof(float);
    DataCopy(d, TileUb<float>(kDecay + bank * kDim * sizeof(float) + remoteHalf, kLaneRows), kLaneRows);
}

__aicore__ inline void PrepareOperands(const Inputs& input, Original& original, GM_ADDR scratch,
                                       uint32_t task, uint32_t pair, uint32_t rows)
{
    const uint32_t lane = original.subid;
    const uint32_t bank = pair % kOperandBanks;
    Mark(input, task, pair, 11 + 2 * lane);
    if (bank == 0) {
        BuildKFactors<0>(original);
    } else {
        BuildKFactors<1>(original);
    }
    ExportOperands(original, DecaySlot(scratch, pair, lane), pair, rows, bank);
    AscendC::PipeBarrier<PIPE_ALL>();
    Mark(input, task, pair, 12 + 2 * lane);
}

__aicore__ inline void ExportInverse(Original& original, uint32_t bank)
{
    using namespace AscendC;
    original.template vec_tril<false>(
        nullptr, RawUb(kTriangular), RawUb(kExchangeUb),
        nullptr, RawUb(kBeta + bank * 256 - original.subid * kRows * sizeof(float)));
    PipeBarrier<PIPE_ALL>();
    original.vec_tril_inv1(RawUb(kInvTmp), RawUb(kTriangular));
    PipeBarrier<PIPE_ALL>();
    original.vec_tril_inv2(RawUb(kInverse), RawUb(kInvTmp), RawUb(kTriangular));
    PipeBarrier<PIPE_ALL>();
    original.template ub_l1_single<kRows, kRows, sizeof(bfloat16_t)>(
        Ub(kInverse), L1b(InverseL1(original.subid)));
}

__aicore__ inline void PublishStateL1(const AscendC::LocalTensor<float>& state, uint32_t lane)
{
    using namespace AscendC;
    constexpr uint16_t lanes = 16;
    constexpr uint16_t rowsPerHalf = kLaneRows / 2;
    constexpr uint32_t laneElements = kLaneRows * kWidth;
    static_assert(kRhsTile + laneElements * sizeof(bfloat16_t) <= kNextUUb);
    auto bf16 = TileUb<bfloat16_t>(kRhsTile, laneElements);
    auto nz = L1b(kStateL1).ReinterpretCast<bfloat16_t>();
    for (uint32_t row = 0; row < kLaneRows; row += rowsPerHalf) {
        Cast(bf16[row * kWidth], state[row * kWidth], RoundMode::CAST_RINT, rowsPerHalf * kWidth);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        // Each burst gathers one 16-column stripe from ND UB into one block of NZ L1.
        const DataCopyParams params{rowsPerHalf, 1, kWidth / lanes - 1, 0};
        for (uint32_t col = 0; col < kWidth; col += lanes) {
            const uint32_t dst = col * kDim + (lane * kLaneRows + row) * lanes;
            DataCopy(nz[dst], bf16[row * kWidth + col], params);
        }
    }
}

__aicore__ inline void InitState(uint32_t lane)
{
    using namespace AscendC;
    auto data = TileUb<float>(kResidentUb, kLaneRows * kWidth);
    Duplicate(data, 0.0f, kLaneRows * kWidth);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < kLaneRows; ++row) {
        data.SetValue(row * kWidth + kDim + lane * kLaneRows + row, 1.0f);
    }
    PipeBarrier<PIPE_ALL>();
    PublishStateL1(data, lane);
}

__aicore__ inline void MakeCorrection(uint32_t lane, uint32_t uOffset)
{
    using namespace AscendC;
    constexpr uint32_t rows = kRows / 2;
    constexpr uint32_t values = rows * kDim;
    constexpr uint32_t count = rows * kWidth;
    auto z = TileUb<float>(kRhsTile, count);
    auto u16 = TileUb<bfloat16_t>(uOffset, values);
    Muls(z, z, -1.0f, count);
    PipeBarrier<PIPE_V>();
    constexpr uint16_t stripeColumns = 16;
    constexpr uint16_t bf16ColumnsPerReg = 128;
    constexpr uint32_t packOffset = kRhsTile + count * sizeof(float) +
                                    values * sizeof(bfloat16_t);
    static_assert(packOffset + count * sizeof(bfloat16_t) <= kNextUUb);
    auto packed = TileUb<bfloat16_t>(packOffset, count);
    auto* zUb = reinterpret_cast<__ubuf__ float*>(z.GetPhyAddr());
    auto* uUb = reinterpret_cast<__ubuf__ bfloat16_t*>(u16.GetPhyAddr());
    auto* packedUb = reinterpret_cast<__ubuf__ bfloat16_t*>(packed.GetPhyAddr());
    __VEC_SCOPE__ {
        MaskReg fp32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            for (uint16_t block = 0; block < static_cast<uint16_t>(kWidth / bf16ColumnsPerReg); ++block) {
                RegTensor<float> even, odd;
                RegTensor<bfloat16_t> lo, hi;
                const uint16_t ndOffset = row * kWidth + block * bf16ColumnsPerReg;
                const uint16_t nzOffset = block * rows * bf16ColumnsPerReg + row * stripeColumns;
                DataCopy<float, LoadDist::DIST_DINTLV_B32>(even, odd, zUb + ndOffset);
                if (block == 0) {
                    RegTensor<float> evenU, oddU;
                    RegTensor<bfloat16_t> packedU;
                    DataCopy(packedU, uUb + row * kDim);
                    Cast<float, bfloat16_t, bf2fZero>(evenU, packedU, bf16);
                    Cast<float, bfloat16_t, bf2fOne>(oddU, packedU, bf16);
                    Add(even, even, evenU, fp32);
                    Add(odd, odd, oddU, fp32);
                }
                Cast<bfloat16_t, float, f2bfZero>(lo, even, fp32);
                Cast<bfloat16_t, float, f2bfOne>(hi, odd, fp32);
                Or(reinterpret_cast<RegTensor<uint16_t>&>(lo), reinterpret_cast<RegTensor<uint16_t>&>(lo),
                   reinterpret_cast<RegTensor<uint16_t>&>(hi), bf16);
                DataCopy<bfloat16_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(packedUb + nzOffset, lo, rows, bf16);
            }
        }
    }
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    auto l1 = L1b(kCorrectionL1).ReinterpretCast<bfloat16_t>();
    const DataCopyParams write{kWidth / stripeColumns, rows, 0, rows};
    DataCopy(l1[lane * rows * stripeColumns], packed, write);
}

template <bool PublishNext>
__aicore__ inline void UpdateStateAndPack(GM_ADDR decay, uint32_t lane, uint32_t localOffset)
{
    using namespace AscendC;
    constexpr uint16_t columnsPerStripe = 16;
    constexpr uint16_t bf16ColumnsPerReg = 128;
    constexpr uint32_t diagOffset = kRhsTile - kLaneRows * sizeof(float);
    constexpr uint32_t laneElements = kLaneRows * kWidth;
    static_assert(diagOffset >= kDecay + 1024);
    static_assert(kRhsTile + laneElements * sizeof(bfloat16_t) <= kNextUUb);
    auto h = TileUb<float>(kResidentUb, laneElements);
    auto delta = TileUb<float>(kDeltaUb, laneElements);
    auto diag = TileUb<float>(localOffset == 0 ? diagOffset : localOffset, kLaneRows);
    auto packed = TileUb<bfloat16_t>(kRhsTile, laneElements);
    if (localOffset == 0) {
        GlobalTensor<float> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay));
        DataCopy(diag, d, kLaneRows);
        PipeBarrier<PIPE_ALL>();
    }
    auto* hUb = reinterpret_cast<__ubuf__ float*>(h.GetPhyAddr());
    auto* deltaUb = reinterpret_cast<__ubuf__ float*>(delta.GetPhyAddr());
    auto* diagUb = reinterpret_cast<__ubuf__ float*>(diag.GetPhyAddr());
    auto* nzUb = reinterpret_cast<__ubuf__ bfloat16_t*>(packed.GetPhyAddr());
    __VEC_SCOPE__ {
        MaskReg fp32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(kLaneRows); ++row) {
            RegTensor<float> scale;
            DataCopy<float, LoadDist::DIST_BRC_B32>(scale, diagUb + row);
            for (uint16_t block = 0; block < static_cast<uint16_t>(kWidth / bf16ColumnsPerReg); ++block) {
                const uint16_t offset = row * kWidth + block * bf16ColumnsPerReg;
                RegTensor<float> even, odd, evenDelta, oddDelta;
                DataCopy<float, LoadDist::DIST_DINTLV_B32>(even, odd, hUb + offset);
                DataCopy<float, LoadDist::DIST_DINTLV_B32>(evenDelta, oddDelta, deltaUb + offset);
                Mul(even, even, scale, fp32);
                Mul(odd, odd, scale, fp32);
                Add(even, even, evenDelta, fp32);
                Add(odd, odd, oddDelta, fp32);
                DataCopy<float, StoreDist::DIST_INTLV_B32>(hUb + offset, even, odd, fp32);
                if constexpr (PublishNext) {
                    RegTensor<bfloat16_t> lo, hi;
                    Cast<bfloat16_t, float, f2bfZero>(lo, even, fp32);
                    Cast<bfloat16_t, float, f2bfOne>(hi, odd, fp32);
                    Or(reinterpret_cast<RegTensor<uint16_t>&>(lo), reinterpret_cast<RegTensor<uint16_t>&>(lo),
                       reinterpret_cast<RegTensor<uint16_t>&>(hi), bf16);
                    const uint16_t nzOffset = block * kLaneRows * bf16ColumnsPerReg + row * columnsPerStripe;
                    DataCopy<bfloat16_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        nzUb + nzOffset, lo, kLaneRows, bf16);
                }
            }
        }
    }
    if constexpr (PublishNext) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        auto nz = L1b(kStateL1).ReinterpretCast<bfloat16_t>();
        const DataCopyParams move{kWidth / columnsPerStripe, kLaneRows, 0, kLaneRows};
        DataCopy(nz[lane * kLaneRows * columnsPerStripe], packed, move);
    }
}

__aicore__ inline void StoreState(GM_ADDR decay, uint32_t lane, bool last, uint32_t localOffset)
{
    if (last) {
        UpdateStateAndPack<false>(decay, lane, localOffset);
    } else {
        UpdateStateAndPack<true>(decay, lane, localOffset);
    }
}

__aicore__ inline void ExportAffineMatrix(uint32_t lane)
{
    using namespace AscendC;
    constexpr uint32_t stripe = 32 / sizeof(float);
    constexpr uint32_t stripes = kDim / stripe;
    auto state = TileUb<float>(kResidentUb, kLaneRows * kWidth);
    auto packed = TileUb<float>(0, kLaneRows * kDim);
    const DataCopyParams stripeCopy{kLaneRows, 1, kWidth / stripe - 1, 0};
    for (uint32_t col = 0; col < stripes; ++col) {
        DataCopy(packed[col * kLaneRows * stripe], state[kDim + col * stripe], stripeCopy);
    }
    PipeBarrier<PIPE_ALL>();
    auto dst = L1b(kAffineMatrixL1).ReinterpretCast<float>();
    const DataCopyParams write{stripes, kLaneRows, 0, kLaneRows};
    DataCopy(dst[lane * kLaneRows * stripe], packed, write);
}

__aicore__ inline void ExportState(GM_ADDR state, uint32_t lane)
{
    using namespace AscendC;
    GlobalTensor<float> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));
    PipeBarrier<PIPE_ALL>();
    DataCopy(output[lane * kLaneRows * kWidth], TileUb<float>(kResidentUb, kLaneRows * kWidth), kLaneRows * kWidth);
    PipeBarrier<PIPE_ALL>();
}

template<typename Offset = int64_t, bool LogAlpha = false>
__aicore__ inline void Run(const Inputs& input, const Schedule& schedule)
{
    using namespace AscendC;
    const uint32_t group = GetBlockIdx() / GetSubBlockNum() - schedule.controlGroups;
    const uint32_t lane = GetSubBlockIdx();
    const bool bound = schedule.boundTask < schedule.tasks;
    // Bound callers provide a single worker's scratch.
    GM_ADDR scratch = input.scratch + (bound ? 0 : group * schedule.scratchStride);
    Original original;
    union { float f; uint32_t u; } gateScale{-5.0f}, epsilon{1.0e-6f};
    // BindTask forms views for the forward-only inputs and outputs as well.
    // Alias those unused views to valid tensors with the expected dimensions.
    original.Init(0, 0, gateScale.u, input.alpha, input.gate, input.key, input.key, input.value,
                  input.beta, input.bias, input.value, input.value, nullptr, nullptr, 0, 0,
                  schedule.tokens, 1, schedule.tasks, schedule.heads, schedule.groups,
                  2, reinterpret_cast<uint64_t>(schedule.offsets), epsilon.u, 3);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    const uint32_t firstTask = bound ? schedule.boundTask : schedule.firstTask + group;
    const uint32_t taskEnd = bound ? schedule.boundTask + 1 : schedule.tasks;
    for (uint32_t task = firstTask; task < taskEnd; task += schedule.groups) {
        original.template BindTaskVarlen<Offset, LogAlpha>(task);
        const uint32_t rows = original.seq_len;
        const uint32_t chunks = (rows + kRows - 1) / kRows;
        const uint32_t pairs = (rows + kRowsPerPair - 1) / kRowsPerPair;
        GM_ADDR state = input.payload == nullptr ? nullptr :
            input.payload + static_cast<uint64_t>(task - schedule.payloadBase) * kPayload * sizeof(float);
        InitState(lane);
        PipeBarrier<PIPE_ALL>();
        Mark(input, task, 0, lane * 2);
        Prepare<0>(original);
        PipeBarrier<PIPE_ALL>();
        Mark(input, task, 0, lane * 2 + 1);

        PrepareOperands(input, original, scratch, task, 0, rows);
        for (uint32_t pair = 0; pair < pairs; ++pair) {
            const uint32_t bank = pair % kOperandBanks;
            // Use the score L1 region only after the previous pair's state has
            // drained. The next pair's GM operands already have their own slot.
            CopyScoreFactorsToL1(original);
            SendCube<kOperandsReady>();
            WaitCube<kScoresReady>();
            ExportInverse(original, bank);
            SendCube<kInverseReady>();
            if (pair + 1 < pairs) {
                PipeBarrier<PIPE_ALL>();
                Mark(input, task, pair + 1, lane * 2);
                if (bank == 0) {
                    Prepare<1>(original);
                } else {
                    Prepare<0>(original);
                }
                PipeBarrier<PIPE_ALL>();
                Mark(input, task, pair + 1, lane * 2 + 1);
            }
            for (uint32_t chunk = 0; chunk < PairChunks(chunks, pair); ++chunk) {
                WaitCube<kProductsReady>();
                MakeCorrection(lane, chunk == 0 ? kUUb : kNextUUb);
                SendCube<kCorrectionReady>();
                if (chunk == 0 && pair + 1 < pairs) {
                    // Current correction shares UB scratch with K operand packing.
                    // Drain the L1 store before the next pair reuses that scratch.
                    PipeBarrier<PIPE_ALL>();
                    PrepareOperands(input, original, scratch, task, pair + 1, rows);
                    PipeBarrier<PIPE_ALL>();
                    SendCube<kDeltaSlotFree>();
                }
                const bool isLast = 2 * pair + chunk + 1 == chunks;
                WaitCube<kResultReady>();
                const uint32_t localDecay = chunk == lane
                    ? kDecay + bank * kDim * sizeof(float) + lane * kLaneRows * sizeof(float) : 0;
                StoreState(DecaySlot(scratch, pair, chunk), lane, isLast, localDecay);
                if (isLast) {
                    if (state == nullptr) {
                        ExportAffineMatrix(lane);
                    } else {
                        ExportState(state, lane);
                    }
                }
                PipeBarrier<PIPE_ALL>();
                SendCube<kStateStored>();
            }
            PipeBarrier<PIPE_ALL>();
            Mark(input, task, pair, 9 + lane);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
}
#endif
} // namespace KcpSummary32
