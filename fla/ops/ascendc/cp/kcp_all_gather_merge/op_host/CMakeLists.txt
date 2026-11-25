file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../merge_fwd_bwd_kernel/op_kernel/
    DESTINATION ${CMAKE_BINARY_DIR}/kcp_merge_reference
    FILES_MATCHING PATTERN "*.h")
include_directories(${CMAKE_BINARY_DIR})
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE kcp_all_gather_merge_def.cpp)
endif()
add_modules_sources(OPTYPE kcp_all_gather_merge ACLNNTYPE aclnn_inner)
if(TARGET cust_opapi)
    # The generated Inner entry points implement the manual public L2 API;
    # exporting them would create a second, undocumented public ABI.
    set(_kcp_exports ${CMAKE_CURRENT_SOURCE_DIR}/kcp_private.map)
    target_link_options(cust_opapi PRIVATE "-Wl,--version-script=${_kcp_exports}")
    set_property(TARGET cust_opapi APPEND PROPERTY LINK_DEPENDS ${_kcp_exports})
endif()
set(_kcp_kernel_options --cce-auto-sync=off -Wno-deprecated-declarations -I${CMAKE_BINARY_DIR})
if("$ENV{FLA_KCP_A5_PROGRESS_PROBE}" MATCHES "^[123]$")
    list(APPEND _kcp_kernel_options "-DFLA_KCP_A5_PROGRESS_PROBE=$ENV{FLA_KCP_A5_PROGRESS_PROBE}")
endif()
add_ops_compile_options(OP_NAME KcpAllGatherMerge OPTIONS ${_kcp_kernel_options})
# Kernel packaging isolates op directories. Make the read-only Merge headers
# available in the native prototype's isolated binary build directory too.
if(ENABLE_PACKAGE)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/kcp_merge_reference
        DESTINATION ${IMPL_INSTALL_DIR}/kcp_all_gather_merge)
endif()
