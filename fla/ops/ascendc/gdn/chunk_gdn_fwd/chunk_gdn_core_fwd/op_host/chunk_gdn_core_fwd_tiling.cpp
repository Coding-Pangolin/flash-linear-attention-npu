/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gdn_core_fwd_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <register/op_impl_registry.h>

namespace optiling {

ge::graphStatus Tiling4ChunkGdnCoreFwdArch32(gert::TilingContext *context);
ge::graphStatus Tiling4ChunkGdnCoreFwdArch35(gert::TilingContext *context);

ge::graphStatus Tiling4ChunkGdnCoreFwd(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGdnCoreFwd", "Invalid tiling context."),
                return ge::GRAPH_FAILED);
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return Tiling4ChunkGdnCoreFwdArch35(context);
    }
    return Tiling4ChunkGdnCoreFwdArch32(context);
}

ge::graphStatus TilingPrepareForChunkGdnCoreFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGdnCoreFwd)
    .Tiling(Tiling4ChunkGdnCoreFwd)
    .TilingParse<ChunkGdnCoreFwdCompileInfo>(TilingPrepareForChunkGdnCoreFwd);

} // namespace optiling
