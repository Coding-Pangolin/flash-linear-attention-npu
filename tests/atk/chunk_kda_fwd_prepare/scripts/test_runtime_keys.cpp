// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include <cassert>
#include <cstddef>
#include <set>
#include <iostream>
#include "chunk_kda_fwd_prepare_runtime_key.h"
#include "chunk_kda_fwd_prepare_tiling_key.h"

int main()
{
    using namespace KdaPrepare;
    std::set<uint64_t> keys;
    for (auto gate : {10U, 30U}) {
        for (auto beta : {10U, 30U}) {
            const auto key = RuntimeKey::Encode(gate, beta, PreparePipeline::Generic);
            assert(keys.insert(key).second);
            std::cout << "generic " << gate << ' ' << beta << ' ' << key << '\n';
        }
        const auto key = RuntimeKey::Encode(gate,10,PreparePipeline::Target);
        assert(keys.insert(key).second);
        std::cout << "target " << gate << " 10 " << key << '\n';
    }
    assert(keys.size()==6);
    const std::set<uint64_t> expected{2570,2590,7690,7710,68106,68126};
    assert(keys==expected);
    // Independently encode the old model/target contracts for a same-source
    // performance control build. This is a restricted development baseline.
    for (uint64_t gate : {10U,30U}) {
        for (uint64_t gm : {1U,2U}) {
            std::cout << "legacy-model " << GET_TPL_TILING_KEY(
                gate,10,1,1,gm,1,gm==2?1ULL:0ULL,2) << '\n';
        }
        std::cout << "legacy-target " << GET_TPL_TILING_KEY(
            gate,10,1,1,0,1,1,2) << '\n';
    }
    RuntimeExpDomain runtime;
    for (bool exp2 : {false,true}) {
        runtime.Init(exp2);
        for (float bound : {-126.F,120.F,-80.F,80.F}) {
            assert(runtime.StoredBound(bound) == (exp2
                ? ExpDomainTraits<true>::StoredBound(bound)
                : ExpDomainTraits<false>::StoredBound(bound)));
        }
        assert(runtime.stepScale == (exp2 ? ExpDomainTraits<true>::stepScale : ExpDomainTraits<false>::stepScale));
        assert(runtime.expInputScale == (exp2 ? ExpDomainTraits<true>::expInputScale : ExpDomainTraits<false>::expInputScale));
    }
    static_assert(PrepareTargetPolicy::arch22FastLeaf);
    static_assert(!PrepareTargetPolicy::safeGate);
    static_assert(!PrepareRuntimePolicy::arch22FastLeaf);
    struct Choice { uint32_t normMode=0,betaMode=0,outputMode=0; } choice;
    assert(RuntimeMath<PrepareRuntimePolicy>::Norm(choice)==QkNormMode::Identity);
    assert(RuntimeMath<PrepareRuntimePolicy>::Beta(choice)==BetaMode::Raw);
    assert(RuntimeMath<PrepareRuntimePolicy>::Output(choice)==OutputMode::None);
    assert(RuntimeMath<PrepareCommonMathPolicy>::Norm(choice)==QkNormMode::L2);
    assert(RuntimeMath<PrepareCommonMathPolicy>::Beta(choice)==BetaMode::Sigmoid);
    assert(RuntimeMath<PrepareCommonMathPolicy>::Output(choice)==OutputMode::Save);
}
