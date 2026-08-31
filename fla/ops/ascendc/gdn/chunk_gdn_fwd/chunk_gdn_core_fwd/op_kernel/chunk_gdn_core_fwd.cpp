/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "internal/arch35/chunk_gdn_core_fwd_arch35.cpp"
#else
#include "internal/arch32/chunk_gdn_core_fwd_arch32.cpp"
#endif
