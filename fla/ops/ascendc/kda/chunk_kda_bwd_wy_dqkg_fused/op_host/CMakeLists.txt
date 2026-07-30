add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_wy_dqkg_fused_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_wy_dqkg_fused ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaBwdWyDqkgFused
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -g
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaBwdWyDqkgFused
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
