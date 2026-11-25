cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(kcp_ipc_chunk32 LANGUAGES ASC CXX)

add_library(kcp_ipc_chunk32 SHARED ipc_kcp_fwd.asc)
target_include_directories(kcp_ipc_chunk32 PRIVATE device)
target_compile_options(kcp_ipc_chunk32 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510;--cce-auto-sync=off>)
target_compile_definitions(kcp_ipc_chunk32 PRIVATE
    KCP_IPC_AFFINE_RELAY=1 KCP_IPC_AFFINE_RELAY_ADVANCE=1)
target_link_directories(kcp_ipc_chunk32 PRIVATE "$ENV{ASCEND_HOME_PATH}/lib64")
target_link_libraries(kcp_ipc_chunk32 PRIVATE ascendcl)
