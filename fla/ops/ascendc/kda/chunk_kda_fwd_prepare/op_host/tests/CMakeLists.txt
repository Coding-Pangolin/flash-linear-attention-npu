# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------

add_executable(
    chunk_kda_fwd_prepare_tiling_processor_test
    chunk_kda_fwd_prepare_tiling_processor_test.cpp
)
target_compile_features(
    chunk_kda_fwd_prepare_tiling_processor_test
    PRIVATE cxx_std_17
)

add_test(
    NAME chunk_kda_fwd_prepare_tiling_processor_test
    COMMAND chunk_kda_fwd_prepare_tiling_processor_test
)

# Portable host-side capability contracts. SDK-dependent key/serializer checks
# remain standalone sources in this directory and use the installed CANN SDK.
foreach(PREPARE_CONTRACT coarse_policy relay_tiling target_tiling)
    set(PREPARE_TEST_NAME chunk_kda_fwd_prepare_${PREPARE_CONTRACT}_test)
    add_executable(${PREPARE_TEST_NAME} test_${PREPARE_CONTRACT}.cpp)
    target_compile_features(${PREPARE_TEST_NAME} PRIVATE cxx_std_17)
    target_include_directories(${PREPARE_TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
    add_test(NAME ${PREPARE_TEST_NAME} COMMAND ${PREPARE_TEST_NAME})
endforeach()
