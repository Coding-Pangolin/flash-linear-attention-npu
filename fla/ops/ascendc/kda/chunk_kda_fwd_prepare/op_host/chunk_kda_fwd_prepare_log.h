/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#ifndef CHUNK_KDA_FWD_PREPARE_LOG_H
#define CHUNK_KDA_FWD_PREPARE_LOG_H

#include "op_common/log/log.h"

namespace kda_prepare {

inline void ReportInvalidArgument(const char *opName, const char *parameter,
                                  const char *reason)
{
#ifdef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, parameter, reason);
#else
    // CANN 9.1 lacks the parameter-level logging helpers. Retain the parameter
    // and reason in its supported error-reporting API.
    OP_LOGE(opName, "Parameter %s is invalid. Reason: %s.", parameter, reason);
#endif
}

inline void ReportInvalidValue(const char *opName, const char *parameter,
                               const char *value, const char *reason)
{
#ifdef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, parameter, value, reason);
#else
    OP_LOGE(opName, "Parameter %s has incorrect value %s. Reason: %s.",
            parameter, value, reason);
#endif
}

} // namespace kda_prepare

#endif // CHUNK_KDA_FWD_PREPARE_LOG_H
