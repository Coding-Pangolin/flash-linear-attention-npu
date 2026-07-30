/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Ascend950 / arch35 Vector.
 *
 * Temporary: reuse the proven 910B classic AscendC vector body. The MicroAPI
 * regbase rewrite (chunk_kda_bwd_wy_dqkg_fused_regbase.h + prior arch35 vector)
 * tripped AICore 507015 on board; keep dual-path scaffold (this include) and
 * ArchTag=Ascend950 via common, re-land regbase after masked-load validation.
 */

#ifndef CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_VECTOR_H

#include "../chunk_kda_bwd_wy_dqkg_fused_vector.h"

#endif // CHUNK_KDA_BWD_WY_DQKG_FUSED_ARCH35_VECTOR_H
