// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "hccl/hccl_types.h"
#include "op_common/log/log.h"
#include "../../op_kernel/kcp_all_gather_merge_tiling.h"
#include <string>

namespace optiling {
static ge::graphStatus TilingKcpAllGatherMerge(gert::TilingContext *context)
{
    auto *td = context->GetTilingData<KcpAllGatherMergeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    const auto *shape = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *group = attrs->GetAttrPointer<char>(0);
    const int64_t *size = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *rank = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, group);
    OP_CHECK_NULL_WITH_CONTEXT(context, size);
    OP_CHECK_NULL_WITH_CONTEXT(context, rank);
    const auto &s = shape->GetStorageShape();
    if (s.GetDimNum() != 3 || s.GetDim(0) < 1 || s.GetDim(0) > 16 ||
        s.GetDim(1) != 128 || s.GetDim(2) != 256 || *size < 2 || *size > 8 ||
        *rank < 0 || *rank >= *size || !group[0]) {
        std::string actual = "summary=[";
        for (size_t i = 0; i < s.GetDimNum(); ++i) actual += (i ? "," : "") + std::to_string(s.GetDim(i));
        actual += "],P=" + std::to_string(*size) + ",rank=" + std::to_string(*rank);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "metadata", actual,
            "require summary storage[H,128,256], H=1..16, P=2..8, 0<=rank<P and nonempty group");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
#if defined(FLA_KCP_A5_PROGRESS_PROBE) && FLA_KCP_A5_PROGRESS_PROBE == 3
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510 && (*size != 2 || s.GetDim(0) < 2)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "metadata", "incompatible probe shape",
            "early-merge probe requires CP2 and at least two heads");
        return ge::GRAPH_FAILED;
    }
#endif
    AscendC::Mc2CcTilingConfig cc(group, HCCL_CMD_ALLGATHER, "AllGather=level0:fullmesh");
    uint32_t initStatus, ccStatus;
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        // MC2 tiling engine 6 (ops-transformer Mc2Comm::ENGINE_CCU_SCHED)
        // is a different enum from the device HcclServerType::CCU = 5.
        AscendC::Mc2CcTilingConfig ccu(group, HCCL_CMD_ALLGATHER, "AllGather=level0:fullmesh",
                                     0, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, 6);
        initStatus = ccu.GetTiling(td->mc2InitTiling);
        ccStatus = ccu.GetTiling(td->mc2CcTiling);
    } else {
        initStatus = cc.GetTiling(td->mc2InitTiling);
        ccStatus = cc.GetTiling(td->mc2CcTiling);
    }
    if (initStatus || ccStatus) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "HCCL tiling status",
            std::to_string(initStatus) + "/" + std::to_string(ccStatus),
            "expected 0/0; verify communicator, service and installed CANN support");
        return ge::GRAPH_FAILED;
    }
    const uint64_t sys = platform.GetLibApiWorkSpaceSize();
    td->merge = {*size, s.GetDim(0), 128, 128, *rank, *rank, 1, 1,
                 static_cast<int64_t>(sys), 0};
    uint32_t launchGroups = 1;
#if defined(FLA_KCP_A5_PROGRESS_PROBE)
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        launchGroups = 2;
    }
#endif
    context->SetBlockDim(launchGroups);
    context->SetTilingKey(30);
    auto *ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = sys + 3ULL * s.GetDim(0) * 128ULL * 128ULL * sizeof(float);
    context->GetRawTilingData()->SetDataSize(sizeof(KcpAllGatherMergeTilingData));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(KcpAllGatherMerge).Tiling(TilingKcpAllGatherMerge);
} // namespace optiling
