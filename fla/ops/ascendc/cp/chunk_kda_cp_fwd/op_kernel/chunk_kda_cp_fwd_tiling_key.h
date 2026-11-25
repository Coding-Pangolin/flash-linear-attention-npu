#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ChunkKdaCpFwd,
    ASCENDC_TPL_DTYPE_DECL(D_T, 10),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, 10)),
);
