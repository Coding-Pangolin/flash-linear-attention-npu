add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gdn_core_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gdn_core_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGdnCoreFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation/gdn_core_solve_tri
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/recompute_w_u_fwd/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/chunk_kkt_solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/chunk_scaled_dot_kkt/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch32/operators/recompute_w_u_fwd/op_kernel
)
