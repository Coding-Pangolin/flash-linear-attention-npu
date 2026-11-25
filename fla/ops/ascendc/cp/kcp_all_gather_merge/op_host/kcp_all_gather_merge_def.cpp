// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "register/op_def_registry.h"

namespace ops {
class KcpAllGatherMerge : public OpDef {
public:
    explicit KcpAllGatherMerge(const char *name) : OpDef(name)
    {
        this->Input("summary").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("h").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("h0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("gathered").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("rank_size").AttrType(REQUIRED).Int();
        this->Attr("rank").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend950");
        this->MC2().HcclGroup("group");
    }
};
OP_ADD(KcpAllGatherMerge);
} // namespace ops
