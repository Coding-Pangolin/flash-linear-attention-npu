# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(chunk_kda_fwd_prepare_host_tests LANGUAGES CXX)
enable_testing()

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

target_include_directories(chunk_kda_fwd_prepare_tiling_processor_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
