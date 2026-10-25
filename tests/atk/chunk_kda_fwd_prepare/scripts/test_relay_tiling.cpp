/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_host/chunk_kda_fwd_prepare_tiling_processor.h"
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare_policy.h"

#include <cstdlib>
#include <limits>

int main()
{
    using namespace KdaPrepare;
    using Arch22RelayConfig::Contract;
    using Arch22RelayConfig::Matches;
    constexpr Contract supported{4, 4, false, 1.0e-6F, -5.0F,
                                 0.08838834764831845F};
    static_assert(Matches(supported));
    static_assert(optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES ==
                  Workspace::kArch22SlotStride);
    static_assert(optiling::CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP ==
                  Workspace::kArch22SlotCount);

    // Head grouping and an incomplete final pair do not change relay legality.
    for (uint32_t heads : {1U, 2U, 3U, 4U, 5U, 8U, 96U, 129U}) {
        auto candidate = supported;
        candidate.qkHeads = candidate.valueHeads = heads;
        if (!Matches(candidate)) {
            return EXIT_FAILURE;
        }
    }
    for (auto heads : {0U, 8U}) {
        auto candidate = supported;
        candidate.qkHeads = heads;
        if (Matches(candidate)) {
            return EXIT_FAILURE;
        }
    }
    auto sequenceMajor = supported;
    sequenceMajor.sequenceMajor = true;
    if (Matches(sequenceMajor)) {
        return EXIT_FAILURE;
    }
    // Parameters outside the accepted numerical contract must use fallback.
    float Contract::*const parameters[] = {
        &Contract::epsilon, &Contract::lowerBound, &Contract::scale};
    for (auto field : parameters) {
        for (float value : {0.0F, 1.0F,
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
            auto candidate = supported;
            candidate.*field = value;
            if (Matches(candidate)) {
                return EXIT_FAILURE;
            }
        }
    }

    // Both representations retain the same scheduler allocation for fixed,
    // variable-length and head-split work. The variant needs no extra slot.
    for (bool varlen : {false, true}) {
        for (uint64_t chunks : {1ULL, 24ULL, 2050ULL, 2292ULL}) {
            for (uint64_t heads : {3ULL, 4ULL, 8ULL}) {
                optiling::ChunkKdaFwdPrepareScheduleContext context{
                    1, heads, heads, chunks, chunks, 20, 1024, varlen};
                optiling::ChunkKdaFwdPrepareSchedule schedule;
                if (!optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) ||
                    schedule.workspaceBytes != 1024 +
                        schedule.usedCoreNum * Workspace::kArch22WorkgroupStride) {
                    return EXIT_FAILURE;
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
