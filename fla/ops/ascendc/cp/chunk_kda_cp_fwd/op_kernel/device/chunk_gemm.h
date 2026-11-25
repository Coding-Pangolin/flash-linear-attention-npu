#pragma once

#include "kernel_operator.h"

namespace KcpState {

static constexpr AscendC::FixpipeConfig kFixRowMajorUb{AscendC::CO2Layout::ROW_MAJOR, true};

// Reuse the same L1/L0 tiles across chunk products. Matrix operands are
// padded row-major GM. Summary tiles use wider BF16 operands.
template<uint16_t MaxRows, uint16_t MaxCols, typename Operand>
class ChunkGemmTiles {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        constexpr uint32_t aBytes = MaxRows * 128 * sizeof(Operand);
        constexpr uint32_t bBytes = MaxCols * 128 * sizeof(Operand);
        pipe.InitBuffer(a1_, aBytes);
        pipe.InitBuffer(b1_, bBytes);
        pipe.InitBuffer(a2_, aBytes);
        pipe.InitBuffer(b2_, bBytes);
        pipe.InitBuffer(c_, MaxRows * MaxCols * sizeof(float));
    }

    template<uint16_t Rows, uint16_t Cols, uint16_t Inner, typename T = bfloat16_t, typename Out = float>
    __aicore__ inline void Run(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR dst, uint16_t lhsStride = Inner)
    {
        RunImpl<Rows, Cols, Inner, T, Out, false>(lhs, rhs, dst, lhsStride, 0);
    }

    template<uint16_t Rows, uint16_t Cols, uint16_t Inner>
    __aicore__ inline void RunToUb(GM_ADDR lhs, GM_ADDR rhs, uint32_t ubOffset, uint16_t lhsStride)
    {
        RunImpl<Rows, Cols, Inner, bfloat16_t, float, true>(lhs, rhs, nullptr, lhsStride, ubOffset);
    }

    template<uint16_t Rows, uint16_t Cols, uint16_t Inner>
    __aicore__ inline void ComputeFull(GM_ADDR lhs, GM_ADDR rhs)
    {
        static_assert(Rows <= MaxRows && Cols <= MaxCols);
        RunImpl<Rows, Cols, Inner, bfloat16_t, float, false, true>(lhs, rhs, nullptr, Inner, 0);
    }

    template<uint16_t Rows, uint16_t Cols>
    __aicore__ inline void PublishCols(uint16_t firstCol, uint32_t ubOffset)
    {
        using namespace AscendC;
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fix;
        fix.nSize = Cols;
        fix.mSize = Rows;
        fix.srcStride = Rows;
        fix.dstStride = Cols;
        fix.params.ndNum = 1;
        fix.dualDstCtl = 1;
        fix.subBlockId = false;
        LocalTensor<float> output(TPosition::VECCALC, ubOffset, Rows * Cols / 2);
        Fixpipe<float, float, kFixRowMajorUb>(output, c_.Get<float>()[firstCol * Rows], fix);
    }

private:
    template<uint16_t Rows, uint16_t Cols, uint16_t Inner, typename T, typename Out, bool ToUb, bool ComputeOnly = false>
    __aicore__ inline void RunImpl(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR dst, uint16_t lhsStride, uint32_t ubOffset)
    {
        using namespace AscendC;
        constexpr uint16_t tileRows = Rows < MaxRows ? Rows : MaxRows;
        constexpr uint16_t tileCols = Cols < MaxCols ? Cols : MaxCols;
        static_assert(Rows % tileRows == 0 && Cols % tileCols == 0 && Inner % 32 == 0 && Inner <= 128);
        static_assert(sizeof(T) <= sizeof(Operand));
        constexpr uint16_t kC0 = 32 / sizeof(T);
        GlobalTensor<T> a, b;
        GlobalTensor<Out> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(lhs));
        b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rhs));
        if constexpr (!ToUb && !ComputeOnly) {
            c.SetGlobalBuffer(reinterpret_cast<__gm__ Out*>(dst));
        }
        auto a1 = a1_.Get<T>();
        auto a2 = a2_.Get<T>();
        auto b1 = b1_.Get<T>();
        auto b2 = b2_.Get<T>();
        auto co = c_.Get<float>();
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fix;
        fix.nSize = tileCols;
        fix.mSize = tileRows;
        fix.srcStride = tileRows;
        fix.dstStride = Cols;
        fix.params.ndNum = 1;
        if constexpr (AscendC::IsSameType<Out, bfloat16_t>::value) {
            fix.quantPre = QuantMode_t::F322BF16;
        }
        for (uint16_t row = 0; row < Rows; row += tileRows) {
            for (uint16_t col = 0; col < Cols; col += tileCols) {
                DataCopy(a1, a[row * lhsStride], Nd2NzParams{1, tileRows, Inner, 0, lhsStride, tileRows, 1, 0});
                DataCopy(b1, b[col], Nd2NzParams{1, Inner, tileCols, 0, Cols, Inner, 1, 0});
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LoadData(a2, a1, LoadData2DParamsV2{
                    0, 0, tileRows / 16, Inner / kC0, tileRows / 16, tileRows / 16, false, 0});
                LoadData(b2, b1, LoadData2DParamsV2{
                    0, 0, Inner / 16, tileCols / kC0, Inner / 16, tileCols / 16, true, 0});
                SetFlag<HardEvent::MTE1_M>(0);
                WaitFlag<HardEvent::MTE1_M>(0);
                MmadParams matmul{tileRows, tileCols, Inner, 0, false, true};
                matmul.disableGemv = true;
                Mmad(co, a2, b2, matmul);
                SetFlag<HardEvent::M_FIX>(0);
                WaitFlag<HardEvent::M_FIX>(0);
                if constexpr (ComputeOnly) {
                    return;
                } else if constexpr (ToUb) {
                    static_assert(tileRows == Rows && tileCols == Cols && Rows % 32 == 0);
                    fix.dualDstCtl = 1;
                    fix.subBlockId = false;
                    LocalTensor<float> output(TPosition::VECCALC, ubOffset, Rows * Cols / 2);
                    Fixpipe<float, float, kFixRowMajorUb>(output, co, fix);
                } else {
                    Fixpipe<Out, float, CFG_ROW_MAJOR>(c[row * Cols + col], co, fix);
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    AscendC::TBuf<AscendC::TPosition::A1> a1_;
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> c_;
};

using ChunkGemm = ChunkGemmTiles<32, 32, float>;
} // namespace KcpState
