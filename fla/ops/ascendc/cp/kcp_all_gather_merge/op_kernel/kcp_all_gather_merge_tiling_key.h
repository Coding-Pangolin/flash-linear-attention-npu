// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

// A single explicit FP32 instance; other dtypes are rejected by public L2.
ASCENDC_TPL_ARGS_DECL(KcpAllGatherMerge,
    ASCENDC_TPL_DTYPE_DECL(D_T, 30),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, 30)),
);
