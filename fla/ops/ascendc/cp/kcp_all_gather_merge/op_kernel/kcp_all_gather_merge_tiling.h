// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#pragma once
#include "kernel_tiling/kernel_tiling.h"
#include "kcp_merge_reference/merge_fwd_bwd_kernel_struct.h"

struct KcpAllGatherMergeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MergeFwBwd::MergeFwdBwdKernelTilingData merge;
};
