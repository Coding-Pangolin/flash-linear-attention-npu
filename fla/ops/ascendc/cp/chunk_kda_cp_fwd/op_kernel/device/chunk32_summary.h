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
constexpr uint32_t kPreparedBytes = (kRows * (kWidth + kDim) + kRows * kRows) * 2 + kDim * 4;
constexpr uint32_t kWuBytes = kRows * kWidth * 2;
constexpr uint32_t kOperandBanks = 2;
constexpr uint32_t kShadowBytes = kPayload * sizeof(bfloat16_t);
constexpr uint32_t kScratchBytes = kOperandBanks * 2 * kPreparedBytes + 2 * kWuBytes +
    kShadowBytes;
constexpr uint32_t kExchangeUb = 84 * 1024;
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
};

struct PreparedChunk {
    GM_ADDR rhs;
    GM_ADDR keyResT;
    GM_ADDR inv;
    GM_ADDR decay;
};

__aicore__ inline PreparedChunk GetPreparedChunk(GM_ADDR scratch, uint32_t pair, uint32_t lane)
{
    GM_ADDR rhs = scratch + ((pair % kOperandBanks) * 2 + lane) * kPreparedBytes;
    GM_ADDR keyResT = rhs + kRows * kWidth * 2;
    GM_ADDR inv = keyResT + kDim * kRows * 2;
    return {rhs, keyResT, inv, inv + kRows * kRows * 2};
}

struct Products {
    GM_ADDR wu;
    GM_ADDR stateBf16;
    GM_ADDR correction;
};

__aicore__ inline Products GetProducts(GM_ADDR scratch)
{
    GM_ADDR wu = scratch + kOperandBanks * 2 * kPreparedBytes;
    GM_ADDR stateBf16 = wu + kWuBytes;
    GM_ADDR correction = stateBf16 + kShadowBytes;
    return {wu, stateBf16, correction};
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
template<typename Offset = int64_t, bool LogAlpha = false>
__aicore__ inline void Run(const Inputs& input, const Schedule& schedule)
{
    const uint32_t group = AscendC::GetBlockIdx() - schedule.controlGroups;
    auto* offsets = reinterpret_cast<__gm__ const Offset*>(schedule.offsets);
    GM_ADDR scratch = input.scratch + group * schedule.scratchStride;
    const auto tmp = GetProducts(scratch);
    Kernel<bfloat16_t> original;
    original.Init();
    AscendC::TPipe pipe;
    KcpState::ChunkGemmTiles<128, 256, bfloat16_t> gemm;
    gemm.Init(pipe);

    for (uint32_t task = schedule.firstTask + group; task < schedule.tasks; task += schedule.groups) {
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
            for (uint32_t chunk = 0; chunk < PairChunks(chunks, pair); ++chunk) {
                const auto f = GetPreparedChunk(scratch, pair, chunk);
                gemm.Run<kRows, kWidth, kRows, bfloat16_t, bfloat16_t>(f.inv, f.rhs, tmp.wu);
                // W occupies the first half of each packed W/U row. The inverse
                // stage has released the UB slot for the two projection halves.
                gemm.RunToUb<kRows, kWidth, kDim>(tmp.wu, tmp.stateBf16, kExchangeUb, kWidth);
                if (chunk == 0) {
                    Mark(input, task, pair, 7);
                }
                SendLanes<kProductsReady>();
                WaitLanes<kCorrectionReady>();
                if (chunk == 0) Mark(input, task, pair, 15);
                // Keep the full Delta MMAD and reuse one UB exchange area for E and M.
                gemm.ComputeFull<kDim, kWidth, kRows>(f.keyResT, tmp.correction);
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
constexpr uint32_t kResidentUb = 120 * KiB;
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

__aicore__ inline void ExportOperands(Original& original, const PreparedChunk& dest, uint32_t pair,
                                      uint32_t rows, uint32_t bank)
{
    using namespace AscendC;
    const uint32_t lane = original.subid;
    const uint32_t start = pair * kRowsPerPair + lane * kRows;
    const uint32_t valid = start >= rows ? 0 : (rows - start < kRows ? rows - start : kRows);
    constexpr uint32_t ndElements = kRows * kDim;
    constexpr uint32_t bfBytes = ndElements * sizeof(bfloat16_t);
    constexpr uint32_t packOffset = kRhsTile + 2 * bfBytes;
    constexpr uint32_t tile = 16;
    static_assert(packOffset + 2 * bfBytes <= kResidentUb);
    auto kNd = TileUb<bfloat16_t>(kRhsTile, ndElements);
    auto vNd = TileUb<bfloat16_t>(kRhsTile + bfBytes, ndElements);
    auto both = TileUb<bfloat16_t>(packOffset, kRows * kWidth);
    auto beta = TileUb<float>(kBeta + bank * 256, kRows);
    GlobalTensor<bfloat16_t> rhs;
    GlobalTensor<uint16_t> kt;
    GlobalTensor<float> d;
    rhs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dest.rhs));
    kt.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(dest.keyResT));
    d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dest.decay));

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
                DataCopy(out + row * kWidth + side * kDim, lo, all);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(rhs, both, kRows * kWidth);

    // NZ stores each 16-column block with a stride of 33 token rows.
    // Transpose its two 16x16 tiles in UB and assemble the ND K-by-token matrix.
    auto nz = TileUb<uint16_t>(kRes, 33 * kDim);
    auto tiles = kNd.ReinterpretCast<uint16_t>();
    auto transposed = vNd.ReinterpretCast<uint16_t>();
    for (uint32_t col = 0; col < kDim / tile; ++col) {
        for (uint32_t piece = 0; piece < kRows / tile; ++piece) {
            const uint32_t block = col * kRows / tile + piece;
            Transpose(tiles[block * tile * tile], nz[(col * 33 + piece * tile) * tile]);
        }
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t col = 0; col < kDim / tile; ++col) {
        for (uint32_t piece = 0; piece < kRows / tile; ++piece) {
            const uint32_t block = col * kRows / tile + piece;
            DataCopy(transposed[col * tile * kRows + piece * tile], tiles[block * tile * tile],
                     DataCopyParams{tile, 1, 0, 1});
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(kt, transposed, ndElements);
    DataCopy(d, TileUb<float>(kDecay + bank * 512, kDim), kDim);
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
    ExportOperands(original, GetPreparedChunk(scratch, pair, lane), pair, rows, bank);
    AscendC::PipeBarrier<PIPE_ALL>();
    Mark(input, task, pair, 12 + 2 * lane);
}

__aicore__ inline void ExportInverse(Original& original, GM_ADDR output, uint32_t bank)
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
    auto nd = TileUb<bfloat16_t>(kRhsTile, kRows * kRows);
    UnpackNz<bfloat16_t, kRows, kRows>(nd, TileUb<bfloat16_t>(kInverse, kRows * kRows));
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<bfloat16_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    DataCopy(out, nd, kRows * kRows);
}

__aicore__ inline void InitState(GM_ADDR shadow, uint32_t lane)
{
    using namespace AscendC;
    auto data = TileUb<float>(kResidentUb, kLaneRows * kWidth);
    Duplicate(data, 0.0f, kLaneRows * kWidth);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < kLaneRows; ++row) {
        data.SetValue(row * kWidth + kDim + lane * kLaneRows + row, 1.0f);
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<bfloat16_t> initial;
    initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(shadow));
    constexpr uint32_t count = kStateRowsPerTile * kWidth;
    auto cast = TileUb<bfloat16_t>(kRhsTile, count);
    for (uint32_t first = 0; first < kLaneRows * kWidth; first += count) {
        Cast(cast, data[first], RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_ALL>();
        DataCopy(initial[lane * kLaneRows * kWidth + first], cast, count);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void MakeCorrection(const Products& tmp, uint32_t lane)
{
    using namespace AscendC;
    constexpr uint32_t rows = kRows / 2;
    constexpr uint32_t values = rows * kDim;
    constexpr uint32_t count = rows * kWidth;
    constexpr uint16_t bfRowBlocks = kDim * sizeof(bfloat16_t) / 32;
    const uint32_t first = lane * count;
    GlobalTensor<bfloat16_t> wu, correction;
    wu.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.wu));
    correction.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.correction));
    auto z = TileUb<float>(kRhsTile, count);
    auto u16 = TileUb<bfloat16_t>(kRhsTile + count * sizeof(float), values);
    auto u = TileUb<float>(kRhsTile + count * sizeof(float) + values * sizeof(bfloat16_t), values);
    DataCopy(u16, wu[first + kDim], DataCopyParams{rows, bfRowBlocks, bfRowBlocks, 0});
    PipeBarrier<PIPE_ALL>();
    Muls(z, z, -1.0f, count);
    Cast(u, u16, RoundMode::CAST_NONE, values);
    PipeBarrier<PIPE_V>();
    // Update E with U - W@E. The M block receives -W@M.
    for (uint32_t i = 0; i < rows; ++i) {
        Add(z[i * kWidth], z[i * kWidth], u[i * kDim], kDim);
    }
    PipeBarrier<PIPE_V>();
    // The packed BF16 result can reuse the U buffers after the additions.
    auto cast = TileUb<bfloat16_t>(kRhsTile + count * sizeof(float), count);
    Cast(cast, z, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_ALL>();
    DataCopy(correction[first], cast, count);
}

__aicore__ inline void StoreState(
    const Products& tmp, GM_ADDR decay, uint32_t lane, bool last)
{
    using namespace AscendC;
    constexpr uint32_t count = kStateRowsPerTile * kWidth;
    constexpr uint32_t diagOffset = kRhsTile + count * sizeof(bfloat16_t);
    static_assert(diagOffset + kLaneRows * sizeof(float) <= kResidentUb);
    auto h = TileUb<float>(kResidentUb, kLaneRows * kWidth);
    auto delta = TileUb<float>(kDeltaUb, kLaneRows * kWidth);
    auto diag = TileUb<float>(diagOffset, kLaneRows);
    GlobalTensor<float> d;
    d.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(decay));
    DataCopy(diag, d[lane * kLaneRows], kLaneRows);
    PipeBarrier<PIPE_ALL>();
    auto* hUb = reinterpret_cast<__ubuf__ float*>(h.GetPhyAddr());
    auto* deltaUb = reinterpret_cast<__ubuf__ float*>(delta.GetPhyAddr());
    auto* diagUb = reinterpret_cast<__ubuf__ float*>(diag.GetPhyAddr());
    __VEC_SCOPE__ {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        constexpr uint16_t lanes = 64;
        for (uint16_t i = 0; i < kLaneRows; ++i) {
            RegTensor<float> decayReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(decayReg, diagUb + i);
            for (uint16_t j = 0; j < kWidth; j += lanes) {
                const uint16_t offset = i * kWidth + j;
                RegTensor<float> prev, change;
                DataCopy(prev, hUb + offset);
                DataCopy(change, deltaUb + offset);
                Mul(prev, prev, decayReg, mask);
                Add(prev, prev, change, mask);
                DataCopy(hUb + offset, prev, mask);
            }
        }
    }
    PipeBarrier<PIPE_V>();
    if (!last) {
        GlobalTensor<bfloat16_t> shadow;
        shadow.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tmp.stateBf16));
        auto cast = TileUb<bfloat16_t>(kRhsTile, count);
        for (uint32_t row = 0; row < kLaneRows; row += kStateRowsPerTile) {
            Cast(cast, h[row * kWidth], RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(shadow[(lane * kLaneRows + row) * kWidth], cast, count);
            PipeBarrier<PIPE_ALL>();
        }
    }
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
    GM_ADDR scratch = input.scratch + group * schedule.scratchStride;
    const auto tmp = GetProducts(scratch);
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
    for (uint32_t task = schedule.firstTask + group; task < schedule.tasks; task += schedule.groups) {
        original.template BindTaskVarlen<Offset, LogAlpha>(task);
        const uint32_t rows = original.seq_len;
        const uint32_t chunks = (rows + kRows - 1) / kRows;
        const uint32_t pairs = (rows + kRowsPerPair - 1) / kRowsPerPair;
        GM_ADDR state = input.payload + static_cast<uint64_t>(task - schedule.payloadBase) * kPayload * sizeof(float);
        InitState(tmp.stateBf16, lane);
        PipeBarrier<PIPE_ALL>();
        Mark(input, task, 0, lane * 2);
        Prepare<0>(original);
        PipeBarrier<PIPE_ALL>();
        Mark(input, task, 0, lane * 2 + 1);

        PrepareOperands(input, original, scratch, task, 0, rows);
        for (uint32_t pair = 0; pair < pairs; ++pair) {
            const uint32_t bank = pair % kOperandBanks;
            const auto f = GetPreparedChunk(scratch, pair, lane);
            // Use the score L1 region only after the previous pair's state has
            // drained. The next pair's GM operands already have their own slot.
            CopyScoreFactorsToL1(original);
            SendCube<kOperandsReady>();
            WaitCube<kScoresReady>();
            ExportInverse(original, f.inv, bank);
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
                MakeCorrection(tmp, lane);
                SendCube<kCorrectionReady>();
                if (chunk == 0 && pair + 1 < pairs) {
                    // Current correction shares UB scratch with K operand packing.
                    // Drain its GM store before the next pair reuses that scratch.
                    PipeBarrier<PIPE_ALL>();
                    PrepareOperands(input, original, scratch, task, pair + 1, rows);
                    PipeBarrier<PIPE_ALL>();
                    SendCube<kDeltaSlotFree>();
                }
                const bool isLast = 2 * pair + chunk + 1 == chunks;
                WaitCube<kResultReady>();
                StoreState(tmp, GetPreparedChunk(scratch, pair, chunk).decay, lane, isLast);
                if (isLast) {
                    ExportState(state, lane);
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
