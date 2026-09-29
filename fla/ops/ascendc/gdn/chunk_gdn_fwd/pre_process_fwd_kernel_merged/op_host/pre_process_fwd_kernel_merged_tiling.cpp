/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_tiling.cpp
 * \brief Host tiling：校验契约 + 填 tiling 结构 + 选 TilingKey（gate 模式）。
 *
 * 契约（docs/api.md）：
 *   k [1,HK,T,K] / w [1,HV,T,K] / u [1,HV,T,V] / v(可选) [1,HV,T,V] / bg(可选) [1,HK,T,K]
 *   g 或 gk 二选一；cu_seqlens 必给（host int 数组 → INT64 tensor），严格递增、0<=cu[0]<cu[-1]<=T
 *   K = V = 128、chunk_size = 64；hm [Nseq,HV,K,V+K] FP32
 */

#include "pre_process_fwd_kernel_merged_tiling.h"

#include <register/op_impl_registry.h>
#include "platform/soc_spec.h"
#include "tiling_base/tiling_templates_registry.h"
#include <cstdlib>   // std::getenv / std::atoll（PPFM_FORCE_COLSPLIT 测试钩子）

namespace optiling {

namespace {
constexpr size_t INPUT_K_IDX = 0;
constexpr size_t INPUT_W_IDX = 1;
constexpr size_t INPUT_U_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_GK_IDX = 4;
constexpr size_t INPUT_BG_IDX = 5;
constexpr size_t INPUT_V_IDX = 6;
constexpr size_t INPUT_SEQLENS_IDX = 7;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

constexpr int64_t PPFM_FIXED_K = 128;
constexpr int64_t PPFM_FIXED_V = 128;
constexpr int64_t PPFM_FIXED_CHUNK = 64;

int64_t DtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN::PPFM_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN::PPFM_DTYPE_FP16;
    }
    return GDN::PPFM_DTYPE_FP32;
}

void PrintTiling(gert::TilingContext *context, const PreProcessFwdKernelMergedTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>> PreProcessFwdKernelMerged tiling <<<<<<<<<<<");
    OP_LOGD(nodeName, "= B:%ld Hk:%ld Hv:%ld hvPerHk:%ld T:%ld K:%ld V:%ld chunkSize:%ld",
            tiling.B, tiling.Hk, tiling.Hv, tiling.hvPerHk, tiling.T, tiling.K, tiling.V, tiling.chunkSize);
    OP_LOGD(nodeName, "= nSeq:%ld gateMode:%ld gateDtype:%ld usedAicNum:%ld taskNum:%ld",
            tiling.nSeq, tiling.gateMode, tiling.gateDtype, tiling.usedAicNum, tiling.taskNum);
}

// Cube(Matmul) 自己的 tiling：v2 的四个矩阵乘只有两种形状
//   ① vTmp = W_c @ bf16(h)、③ T1 = W_c @ bf16(m)：M=BT, N=128, K=128，A 不转置
//   ② dH   = k_c^T @ bf16(v_new)、④ T2 = left^T @ bf16(T1)：M=128, N=128, K=BT，A 转置
bool BuildCubeTiling(const platform_ascendc::PlatformAscendC &platform,
                     GDN::PreProcessFwdKernelMergedTilingData *tiling)
{
    constexpr uint64_t kBT = 64ULL;
    constexpr uint64_t kKv = 128ULL;
    {
        matmul_tiling::MatmulApiTiling mm(platform);
        if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
            mm.EnableBias(false) != 0 ||
            mm.SetShape(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv),
                        static_cast<int32_t>(kKv)) != 0 ||
            mm.SetOrgShape(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv),
                           static_cast<int32_t>(kKv)) != 0 ||
            mm.SetFixSplit(static_cast<int32_t>(kBT), static_cast<int32_t>(kKv), -1) != 0 ||
            mm.SetBufferSpace(-1, -1, 0, -1) != 0 ||
            mm.GetTiling(tiling->cubeNoTrans) == -1) {
            return false;
        }
    }
    {
        matmul_tiling::MatmulApiTiling mm(platform);
        if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, true) != 0 ||
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
            mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
            mm.EnableBias(false) != 0 ||
            mm.SetShape(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv),
                        static_cast<int32_t>(kBT)) != 0 ||
            mm.SetOrgShape(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv),
                           static_cast<int32_t>(kBT)) != 0 ||
            mm.SetFixSplit(static_cast<int32_t>(kKv), static_cast<int32_t>(kKv), -1) != 0 ||
            mm.SetBufferSpace(-1, -1, 0, -1) != 0 ||
            mm.GetTiling(tiling->cubeTransA) == -1) {
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus Tiling4PreProcessFwdKernelMerged(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<PreProcessFwdKernelMergedTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kShapePtr = context->GetInputShape(INPUT_K_IDX);
    auto wShapePtr = context->GetInputShape(INPUT_W_IDX);
    auto uShapePtr = context->GetInputShape(INPUT_U_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, uShapePtr);
    const gert::Shape kShape = kShapePtr->GetStorageShape();
    const gert::Shape uShape = uShapePtr->GetStorageShape();

    const int64_t B = kShape.GetDim(0);
    const int64_t Hk = kShape.GetDim(1);
    const int64_t T = kShape.GetDim(2);
    const int64_t K = kShape.GetDim(3);
    const int64_t Hv = uShape.GetDim(1);
    const int64_t V = uShape.GetDim(3);
    const int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));

    OP_CHECK_IF(B != 1, OP_LOGE(context->GetNodeName(), "B must be 1 (varlen packed), got %ld", B),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(K != PPFM_FIXED_K || V != PPFM_FIXED_V,
                OP_LOGE(context->GetNodeName(), "K/V must be 128, got %ld/%ld", K, V),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(chunkSize != PPFM_FIXED_CHUNK,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64, got %ld", chunkSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Hk <= 0 || Hv <= 0 || Hv % Hk != 0,
                OP_LOGE(context->GetNodeName(), "GVA requires Hv %% Hk == 0, got Hk=%ld Hv=%ld", Hk, Hv),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wShapePtr->GetStorageShape().GetDim(1) != Hv,
                OP_LOGE(context->GetNodeName(), "w head dim must equal Hv"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(uShape.GetDim(0) != 1 || uShape.GetDim(2) != T || uShape.GetDim(3) != V,
                OP_LOGE(context->GetNodeName(), "u must be [1,Hv,T,V] and match k's T"), return ge::GRAPH_FAILED);

    // gate 二选一 + bg 与 gk 配套
    auto gTensor = context->GetOptionalInputTensor(INPUT_G_IDX);
    auto gkTensor = context->GetOptionalInputTensor(INPUT_GK_IDX);
    auto bgTensor = context->GetOptionalInputTensor(INPUT_BG_IDX);
    const bool hasG = gTensor != nullptr;
    const bool hasGk = gkTensor != nullptr;
    const bool hasBg = bgTensor != nullptr;
    OP_CHECK_IF(hasG == hasGk, OP_LOGE(context->GetNodeName(), "exactly one of g / gk must be given"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hasBg && !hasGk, OP_LOGE(context->GetNodeName(), "DPLR requires bg paired with gk"),
                return ge::GRAPH_FAILED);
    const int64_t gateMode = hasBg ? GDN::PPFM_GATE_USE_BG : (hasGk ? GDN::PPFM_GATE_USE_GK : GDN::PPFM_GATE_USE_G);
    const ge::DataType gateDtype = hasG ? gTensor->GetDataType() : gkTensor->GetDataType();

    // cu_seqlens：host int 数组，必给；校验 0 <= cu[0] < ... < cu[-1] <= T
    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    OP_CHECK_IF(cuSeqlensTensor == nullptr,
                OP_LOGE(context->GetNodeName(), "cu_seqlens is required (varlen only)"), return ge::GRAPH_FAILED);
    const int64_t cuNumel = static_cast<int64_t>(cuSeqlensTensor->GetShapeSize());
    OP_CHECK_IF(cuNumel < 2, OP_LOGE(context->GetNodeName(), "cu_seqlens must have >= 2 elements"),
                return ge::GRAPH_FAILED);
    const int64_t *cuData = cuSeqlensTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, cuData);
    for (int64_t i = 0; i < cuNumel; ++i) {
        OP_CHECK_IF(cuData[i] < 0 || cuData[i] > T,
                    OP_LOGE(context->GetNodeName(), "cu_seqlens[%ld]=%ld out of range [0,%ld]", i, cuData[i], T),
                    return ge::GRAPH_FAILED);
        if (i > 0) {
            OP_CHECK_IF(cuData[i] <= cuData[i - 1],
                        OP_LOGE(context->GetNodeName(), "cu_seqlens must be strictly increasing at %ld", i),
                        return ge::GRAPH_FAILED);
        }
    }
    const int64_t nSeq = cuNumel - 1;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t aicNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
    // ---- P5 列块切分：把"链"按列切成 colSplit 份，用来消掉波次量化 ----
    // 单条链的成本 ≈ (a + nt·b)，切 s 份后每份 ≈ (a + nt·b)/s（列维独立且可切）。
    // 总时间 ≈ ceil(N·s / aicNum) × (a + nt·b)/s，只跟波数有关 ⇒ 选 waves(s)/s 最小的 s。
    // 约束：cube 的 N 维（= V/K 列宽 = 128/s）不能太窄，只考虑 s ∈ {1,2}（N=128 / 64）。
    const int64_t hwItems = nSeq * Hv;
    auto wavesOf = [aicNum](int64_t items) {
        return (aicNum > 0) ? (items + aicNum - 1) / aicNum : int64_t{1};
    };
    int64_t colSplit = 1;
    if (aicNum > 0 && wavesOf(hwItems * 2) * 1 < wavesOf(hwItems) * 2) {
        colSplit = 2;
    }
    // 测试/调试钩子：环境变量可强制列块切分因子（1 或 2），用于位级 A/B 与回归。
    if (const char *forceSplit = std::getenv("PPFM_FORCE_COLSPLIT")) {
        const int64_t v = std::atoll(forceSplit);
        if (v == 1 || v == 2) {
            colSplit = v;
        }
    }
    const int64_t taskNum = hwItems * colSplit;

    tiling->B = B;
    tiling->Hk = Hk;
    tiling->Hv = Hv;
    tiling->hvPerHk = Hv / Hk;
    tiling->T = T;
    tiling->K = K;
    tiling->V = V;
    tiling->chunkSize = chunkSize;
    tiling->nSeq = nSeq;
    tiling->gateMode = gateMode;
    tiling->gateDtype = DtypeToEnum(gateDtype);
    tiling->isVariedLen = 1;
    tiling->usedAicNum = (taskNum < aicNum) ? taskNum : aicNum;
    tiling->taskNum = taskNum;
    tiling->colSplit = colSplit;

    OP_CHECK_IF(context->GetRawTilingData() == nullptr ||
                    context->GetRawTilingData()->GetCapacity() <
                        sizeof(GDN::PreProcessFwdKernelMergedTilingData),
                OP_LOGE(context->GetNodeName(), "tiling buffer too small for tcube tiling"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!BuildCubeTiling(ascendcPlatform, tiling),
                OP_LOGE(context->GetNodeName(), "build cube(TCubeTiling) tiling failed"),
                return ge::GRAPH_FAILED);

    context->SetTilingKey(static_cast<uint32_t>(gateMode) + 1U);
    context->SetBlockDim(static_cast<uint32_t>(tiling->usedAicNum));
    context->SetScheduleMode(1);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 系统 workspace + 每个工作项一块 GM 临时区（kernel 侧口径见 op_kernel/..._struct.h）
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() +
                          static_cast<size_t>(tiling->usedAicNum) *
                              static_cast<size_t>(GDN::PPFM_CORE_WS_BYTES);
    PrintTiling(context, *tiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParse4PreProcessFwdKernelMerged(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PreProcessFwdKernelMerged)
    .Tiling(Tiling4PreProcessFwdKernelMerged)
    .TilingParse<PreProcessFwdKernelMergedCompileInfo>(TilingParse4PreProcessFwdKernelMerged);

} // namespace optiling
