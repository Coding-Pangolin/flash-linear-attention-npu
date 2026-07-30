/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Ascend950 MicroAPI (regbase) helpers for ChunkKdaBwdWyDqkgFused vector path.
 * Pattern mirrors causal_conv1d_regbase / prepare_wy_repr_bwd_full_vector.
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_REGBASE_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_REGBASE_H

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

namespace kda_wy_dqkg {
namespace regbase_ops {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);

__aicore__ inline uint16_t CeilDivU16(uint32_t n, uint16_t d)
{
    return static_cast<uint16_t>((n + static_cast<uint32_t>(d) - 1U) / static_cast<uint32_t>(d));
}

// t[i] = exp2(clamp(t[i]))  <=>  exp(clamp(t*ln2))
__aicore__ inline void Exp2InPlaceReg(LocalTensor<float> t, uint32_t count, float ln2, float expMax, float expMin)
{
    if (count == 0) {
        return;
    }
    __ubuf__ float *addr = (__ubuf__ float *)t.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = CeilDivU16(count, VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        MaskReg preg;
        for (uint16_t i = 0; i < loops; ++i) {
            preg = UpdateMask<float>(remain);
            DataCopy(x, addr + i * VL_F32);
            Muls(x, x, ln2, preg);
            Mins(x, x, expMax, preg);
            Maxs(x, x, expMin, preg);
            Exp(x, x, preg);
            DataCopy(addr + i * VL_F32, x, preg);
        }
    }
}

__aicore__ inline void MulInPlaceReg(LocalTensor<float> dst, LocalTensor<float> src0, LocalTensor<float> src1,
                                     uint32_t count)
{
    if (count == 0) {
        return;
    }
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *a = (__ubuf__ float *)src0.GetPhyAddr();
    __ubuf__ float *b = (__ubuf__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = CeilDivU16(count, VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> ra, rb, rc;
        MaskReg preg;
        for (uint16_t i = 0; i < loops; ++i) {
            preg = UpdateMask<float>(remain);
            DataCopy(ra, a + i * VL_F32);
            DataCopy(rb, b + i * VL_F32);
            Mul(rc, ra, rb, preg);
            DataCopy(d + i * VL_F32, rc, preg);
        }
    }
}

__aicore__ inline void AddInPlaceReg(LocalTensor<float> dst, LocalTensor<float> src0, LocalTensor<float> src1,
                                     uint32_t count)
{
    if (count == 0) {
        return;
    }
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *a = (__ubuf__ float *)src0.GetPhyAddr();
    __ubuf__ float *b = (__ubuf__ float *)src1.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = CeilDivU16(count, VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> ra, rb, rc;
        MaskReg preg;
        for (uint16_t i = 0; i < loops; ++i) {
            preg = UpdateMask<float>(remain);
            DataCopy(ra, a + i * VL_F32);
            DataCopy(rb, b + i * VL_F32);
            Add(rc, ra, rb, preg);
            DataCopy(d + i * VL_F32, rc, preg);
        }
    }
}

__aicore__ inline void MulsInPlaceReg(LocalTensor<float> dst, LocalTensor<float> src, float scalar, uint32_t count)
{
    if (count == 0) {
        return;
    }
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *a = (__ubuf__ float *)src.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = CeilDivU16(count, VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> ra, rc;
        MaskReg preg;
        for (uint16_t i = 0; i < loops; ++i) {
            preg = UpdateMask<float>(remain);
            DataCopy(ra, a + i * VL_F32);
            Muls(rc, ra, scalar, preg);
            DataCopy(d + i * VL_F32, rc, preg);
        }
    }
}

__aicore__ inline void DuplicateReg(LocalTensor<float> dst, float value, uint32_t count)
{
    if (count == 0) {
        return;
    }
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    uint32_t remain = count;
    const uint16_t loops = CeilDivU16(count, VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        MaskReg preg;
        for (uint16_t i = 0; i < loops; ++i) {
            preg = UpdateMask<float>(remain);
            Duplicate(x, value, preg);
            DataCopy(d + i * VL_F32, x, preg);
        }
    }
}

// dst[r,c] *= rowVec[r]
__aicore__ inline void RowBroadcastMulInPlaceReg(LocalTensor<float> dst, LocalTensor<float> rowVec, uint32_t rows,
                                                 uint32_t cols)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    __ubuf__ float *dAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *rAddr = (__ubuf__ float *)rowVec.GetPhyAddr();
    const uint16_t colLoops = CeilDivU16(cols, VL_F32);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t remain = cols;
        __VEC_SCOPE__
        {
            RegTensor<float> rowScal, mat, out;
            MaskReg preg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rowScal, rAddr + r);
            for (uint16_t c = 0; c < colLoops; ++c) {
                preg = UpdateMask<float>(remain);
                DataCopy(mat, dAddr + r * cols + c * VL_F32);
                Mul(out, mat, rowScal, preg);
                DataCopy(dAddr + r * cols + c * VL_F32, out, preg);
            }
        }
    }
}

// dst[r,c] *= colVec[c]
__aicore__ inline void ColBroadcastMulInPlaceReg(LocalTensor<float> dst, LocalTensor<float> colVec, uint32_t rows,
                                                 uint32_t cols)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    __ubuf__ float *dAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *cAddr = (__ubuf__ float *)colVec.GetPhyAddr();
    const uint16_t colLoops = CeilDivU16(cols, VL_F32);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t remain = cols;
        __VEC_SCOPE__
        {
            RegTensor<float> col, mat, out;
            MaskReg preg;
            for (uint16_t c = 0; c < colLoops; ++c) {
                preg = UpdateMask<float>(remain);
                DataCopy(col, cAddr + c * VL_F32);
                DataCopy(mat, dAddr + r * cols + c * VL_F32);
                Mul(out, mat, col, preg);
                DataCopy(dAddr + r * cols + c * VL_F32, out, preg);
            }
        }
    }
}

// acc[r] += sum_c mat[r,c]
__aicore__ inline void RowFoldSumAddIntoReg(LocalTensor<float> acc, LocalTensor<float> mat, uint32_t rows,
                                            uint32_t width)
{
    if (rows == 0 || width == 0) {
        return;
    }
    __ubuf__ float *accAddr = (__ubuf__ float *)acc.GetPhyAddr();
    __ubuf__ float *matAddr = (__ubuf__ float *)mat.GetPhyAddr();
    const uint16_t colLoops = CeilDivU16(width, VL_F32);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t remain = width;
        uint32_t storeCnt = 1;
        __VEC_SCOPE__
        {
            RegTensor<float> chunk, partial, total, accReg;
            MaskReg preg;
            MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
            Duplicate(total, 0.0f, maskAll);
            for (uint16_t c = 0; c < colLoops; ++c) {
                preg = UpdateMask<float>(remain);
                DataCopy(chunk, matAddr + r * width + c * VL_F32);
                ReduceSum(partial, chunk, preg);
                Add(total, total, partial, maskAll);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(accReg, accAddr + r);
            Add(accReg, accReg, total, maskAll);
            MaskReg maskOne = UpdateMask<float>(storeCnt);
            DataCopy(accAddr + r, accReg, maskOne);
        }
    }
}

// Dense T→fp32 / fp32→T via classic Cast (MicroAPI unpack layout is VL-sensitive);
// kept here so stage call sites share one A5 entry point.
template <typename T>
__aicore__ inline void CastTToF32Reg(LocalTensor<float> dst, LocalTensor<T> src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    Cast(dst, src, RoundMode::CAST_NONE, count);
}

template <typename T>
__aicore__ inline void CastF32ToTReg(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    Cast(dst, src, RoundMode::CAST_RINT, count);
}

} // namespace regbase_ops
} // namespace kda_wy_dqkg

#endif // CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_REGBASE_H
