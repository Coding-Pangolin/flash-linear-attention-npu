#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ChunkKdaCpFwdTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    uint32_t heads;
    uint32_t tokens;
    uint32_t segments;
    uint32_t workers;
    uint32_t ranks;
    uint32_t rank;
    uint32_t rounds;
    uint32_t stateRelay;
    uint32_t sendTail;
    int32_t firstSource;
    uint32_t saveFinal;
    float scale;
    uint64_t workerOffset;
};

namespace KcpWorkspace {
// Internal host plan value for the two-batch state relay. Tiling records the
// actual batch count and schedule separately.
constexpr int64_t kStateRelayPlan = 3;
constexpr uint32_t kRelayBatchHeads = 16;
constexpr uint64_t kDim = 128;
constexpr uint64_t kStateBytes = kDim * kDim * sizeof(float);
constexpr uint64_t kSummaryBytes = 2 * kStateBytes;
constexpr uint64_t kRelayBatchBytes = kRelayBatchHeads * kStateBytes;
constexpr uint64_t kWorkerBytes = 768 * 1024;

} // namespace KcpWorkspace
