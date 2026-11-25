// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaCpFwd : public OpDef {
public:
    explicit ChunkKdaCpFwd(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("gate").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("a_log").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("offsets").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("state").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("rank_size").AttrType(REQUIRED).Int();
        this->Attr("rank").AttrType(REQUIRED).Int();
        this->Attr("rounds").AttrType(REQUIRED).Int();
        this->Attr("first_source").AttrType(REQUIRED).Int();
        this->Attr("send_tail").AttrType(REQUIRED).Bool(false);
        this->Attr("save_final").AttrType(REQUIRED).Bool(true);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0f);
        this->AICore().AddConfig("ascend950");
        this->MC2().HcclGroup("group");
    }
};
OP_ADD(ChunkKdaCpFwd);
} // namespace ops
