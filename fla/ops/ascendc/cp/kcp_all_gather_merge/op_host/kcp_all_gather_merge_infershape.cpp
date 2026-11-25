// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferKcp(gert::InferShapeContext *context)
{
    const auto *summary = context->GetInputShape(0);
    const auto *h = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    if (!summary || !h || !attrs || summary->GetDimNum() != 3) return ge::GRAPH_FAILED;
    const auto *size = attrs->GetAttrPointer<int64_t>(1);
    if (!size) return ge::GRAPH_FAILED;
    *context->GetOutputShape(0) = *h;
    auto *out = context->GetOutputShape(1);
    out->SetDimNum(4);
    out->SetDim(0, *size);
    for (int i = 0; i < 3; ++i) out->SetDim(i+1, summary->GetDim(i));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeKcp(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(KcpAllGatherMerge).InferShape(InferKcp).InferDataType(DtypeKcp);
} // namespace ops
