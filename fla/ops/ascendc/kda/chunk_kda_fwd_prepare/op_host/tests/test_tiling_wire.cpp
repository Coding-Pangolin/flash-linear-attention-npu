// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#include <cassert>
#include <cstring>
#include <iostream>
#include "chunk_kda_fwd_prepare_tiling.h"
#include "chunk_kda_fwd_prepare_tiling_fields.h"

int main()
{
    // Verify the real CANN serializer, rather than a second handwritten encoder.
    // Distinct field values catch permutations, bool padding and truncated data.
    optiling::ChunkKdaFwdPrepareTilingData host;
    KdaPrepare::PrepareTilingWire expected{};
    uint32_t value = 1;
#define PREPARE_SET_FIELD(T, NAME) \
    expected.NAME = static_cast<T>(value++); host.set_##NAME(expected.NAME)
    CHUNK_KDA_PREPARE_TILING_FIELDS(PREPARE_SET_FIELD);
#undef PREPARE_SET_FIELD
    assert(host.GetDataSize() == sizeof(expected));
    KdaPrepare::PrepareTilingWire actual{};
    host.SaveToBuffer(&actual, sizeof(actual));
#define PREPARE_CHECK_FIELD(T, NAME) assert(actual.NAME == expected.NAME)
    CHUNK_KDA_PREPARE_TILING_FIELDS(PREPARE_CHECK_FIELD);
#undef PREPARE_CHECK_FIELD
    // Check false values separately: casting the distinct values above gives true.
    host.set_isVarLen(false);
    host.set_inputSequenceMajor(false);
    host.set_hasDtBias(false);
    host.set_useExp2(false);
    host.SaveToBuffer(&actual, sizeof(actual));
    assert(!actual.isVarLen && !actual.inputSequenceMajor && !actual.hasDtBias && !actual.useExp2);
    std::cout << "tiling_wire_bytes " << host.GetDataSize() << '\n';
}
