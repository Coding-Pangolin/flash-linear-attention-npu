/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint64_t CHUNK_KDA_FWD_PREPARE_CHUNK_ROWS = 64;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES = 0x1A400;
// Arch22 的 slot 在 "105 KiB 公共区 + 20 KiB AIC relay" 之后追加 32 KiB RHS
// 独立区（K_beta_g/V_beta），使 V6→C7 的交接不再与 V1 payload 复用地址。
// 该值必须与 kernel 侧 Workspace::kArch22SlotStride 保持一致。
// 0x27400 + 16 KiB「q/k 行平方和」relay（AIC 预算、AIV 的 V0 读取）。
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES = 0x2B400;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP = 4;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH35_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;
constexpr uint64_t CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES =
    CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES *
    CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP;

struct ChunkKdaFwdPrepareScheduleContext {
    uint64_t batch = 0;
    uint64_t qkHeadNum = 0;
    uint64_t valueHeadNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t totalVarLenChunks = 0;
    uint64_t aicCoreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    bool isVarLen = false;
    // 调用方未显式识别平台时按较大的 Arch22 slot 保守分配。
    uint64_t workspaceSlotBytes = CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
};

struct ChunkKdaFwdPrepareSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    uint64_t chunkWorkItems = 0;
    uint64_t totalWorkItems = 0;
    uint64_t workspaceBytes = 0;
};

class ChunkKdaFwdPrepareTilingProcessor {
public:
    explicit ChunkKdaFwdPrepareTilingProcessor(
        const ChunkKdaFwdPrepareScheduleContext &context) : context_(context)
    {}

    bool Process(ChunkKdaFwdPrepareSchedule &schedule) const
    {
        if (context_.batch == 0 || context_.qkHeadNum == 0 ||
            context_.valueHeadNum < context_.qkHeadNum ||
            context_.valueHeadNum % context_.qkHeadNum != 0 ||
            context_.aicCoreNum == 0 ||
            (context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES &&
             context_.workspaceSlotBytes !=
                 CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES)) {
            return false;
        }

        uint64_t chunkWorkItems = context_.totalVarLenChunks;
        if (!context_.isVarLen) {
            if (!CheckedMultiply(context_.batch, context_.chunksPerSequence,
                                 chunkWorkItems)) {
                return false;
            }
        }
        if (chunkWorkItems == 0) {
            return false;
        }
        if (chunkWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        // head 切分数 P 直接决定每核的 head-chunk 数：
        //   cost(P) = ceil(chunkWorkItems*P/cores) * (HV/P)
        // 只在"多切一刀能减少最忙核工作量"时才切；切分必须落在完整 GVA
        // QK 头组边界上，保证 q_hat/k_hat 的 owner 语义不变。
        const uint64_t qkGroup =
            context_.valueHeadNum / context_.qkHeadNum;
        uint64_t bestPartitionCount = 1;
        uint64_t bestCost = ~0ULL;
        for (uint64_t p = 1; p <= context_.qkHeadNum && p <= 4; ++p) {
            if (context_.valueHeadNum % p != 0) {
                continue;
            }
            const uint64_t perPartition = context_.valueHeadNum / p;
            if (qkGroup != 0 && perPartition % qkGroup != 0) {
                continue;
            }
            const uint64_t items = chunkWorkItems * p;
            const uint64_t cores =
                items < context_.aicCoreNum ? items : context_.aicCoreNum;
            const uint64_t trips = (items + cores - 1) / cores;
            const uint64_t cost = trips * perPartition;
            if (cost < bestCost) {
                bestCost = cost;
                bestPartitionCount = p;
            }
        }
        const uint64_t partitionCount = bestPartitionCount;
        const uint64_t headsPerPartition =
            context_.valueHeadNum / partitionCount;

        uint64_t totalWorkItems = 0;
        if (!CheckedMultiply(chunkWorkItems, partitionCount, totalWorkItems)) {
            return false;
        }
        if (totalWorkItems > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t usedCoreNum = std::min(context_.aicCoreNum, totalWorkItems);
        uint64_t workgroupBytes = 0;
        uint64_t userWorkspaceBytes = 0;
        if (!CheckedMultiply(context_.workspaceSlotBytes,
                             CHUNK_KDA_FWD_PREPARE_SLOTS_PER_WORKGROUP,
                             workgroupBytes) ||
            !CheckedMultiply(usedCoreNum, workgroupBytes,
                             userWorkspaceBytes) ||
            context_.libApiWorkspaceBytes >
                std::numeric_limits<uint64_t>::max() - userWorkspaceBytes ||
            usedCoreNum > std::numeric_limits<uint32_t>::max() ||
            headsPerPartition > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        schedule.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
        schedule.headsPerPartition = static_cast<uint32_t>(headsPerPartition);
        schedule.chunkWorkItems = chunkWorkItems;
        schedule.totalWorkItems = totalWorkItems;
        schedule.workspaceBytes = context_.libApiWorkspaceBytes + userWorkspaceBytes;
        return true;
    }

private:
    static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
            return false;
        }
        result = lhs * rhs;
        return true;
    }

    const ChunkKdaFwdPrepareScheduleContext &context_;
};

} // namespace optiling

#endif // CHUNK_KDA_FWD_PREPARE_TILING_PROCESSOR_H
