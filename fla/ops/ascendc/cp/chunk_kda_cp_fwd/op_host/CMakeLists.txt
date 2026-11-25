# Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
add_op_to_compiled_list()
set(chunk_kda_cp_fwd_kernel_depends "fla/ops/ascendc/kda/chunk_kda_fwd_fused"
    CACHE STRING "Kernel source dependencies for chunk_kda_cp_fwd" FORCE)
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE chunk_kda_cp_fwd_def.cpp)
endif()
add_modules_sources(OPTYPE chunk_kda_cp_fwd ACLNNTYPE aclnn_inner)
add_ops_compile_options(OP_NAME ChunkKdaCpFwd OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
if(TARGET cust_opapi)
    set(_kcp_exports ${CMAKE_CURRENT_SOURCE_DIR}/chunk_kda_cp_fwd_private.map)
    target_link_options(cust_opapi PRIVATE "-Wl,--version-script=${_kcp_exports}")
    set_property(TARGET cust_opapi APPEND PROPERTY LINK_DEPENDS ${_kcp_exports})
endif()
