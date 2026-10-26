// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include <cassert>
#include <initializer_list>
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_coarse_score.h"

struct Args {
    struct {
        bool inputSequenceMajor = false;
        uint32_t qkHeadNum = 4, valueHeadNum = 4;
        uint32_t normMode = 1, gateMode = 2, outputMode = 2;
        float lowerBound = -5.0F;
    } tiling;
};

int main()
{
    using namespace KdaPrepare;
    for (uint32_t norm : {0U,1U}) for (uint32_t gate : {0U,1U,2U})
    for (uint32_t output : {0U,1U,2U,3U}) for (bool seq : {false,true})
    for (uint32_t hk : {0U,1U,3U,4U,8U})
    for (uint32_t heads : {3U,4U,8U,9U}) for (float bound : {-6.F,-5.F,-1.F,0.F}) {
        Args args;
        auto &t=args.tiling;
        t.normMode=norm; t.gateMode=gate; t.outputMode=output;
        t.inputSequenceMajor=seq; t.qkHeadNum=hk; t.valueHeadNum=heads; t.lowerBound=bound;
        const bool expected=!seq && hk!=0 && heads>=hk && heads%hk==0 && norm==1 && gate==2 &&
            (output==1 || output==2) && bound>=-5.F && bound<0.F;
        assert(Arch22::CoarseScore::Eligible<PrepareRuntimePolicy>(args)==expected);
        // CommonMath is only selected for the public L2/Save contract. Gate
        // remains runtime; folding must not enable coarse score for softplus.
        if (norm==1 && output==2) {
            assert(Arch22::CoarseScore::Eligible<PrepareCommonMathPolicy>(args)==expected);
        }
    }
}
