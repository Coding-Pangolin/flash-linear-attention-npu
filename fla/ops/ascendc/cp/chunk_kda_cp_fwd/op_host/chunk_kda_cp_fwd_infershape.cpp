// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferKcpFwd(gert::InferShapeContext* context)
{
    const auto* q = context->GetInputShape(0);
    const auto* offsets = context->GetInputShape(7);
    const auto* attrs = context->GetAttrs();
    const auto* save = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(6);
    if (!q || !offsets || q->GetDimNum() != 4 || offsets->GetDimNum() != 1 || save == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* out = context->GetOutputShape(0);
    auto* state = context->GetOutputShape(1);
    *out = gert::Shape{1, q->GetDim(2), q->GetDim(1), 128};
    *state = *save ? gert::Shape{offsets->GetDim(0) - 1, q->GetDim(1), 128, 128}
                   : gert::Shape{1, 1, 1, 1};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferKcpFwdDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ChunkKdaCpFwd).InferShape(InferKcpFwd).InferDataType(InferKcpFwdDtype);
} // namespace ops
