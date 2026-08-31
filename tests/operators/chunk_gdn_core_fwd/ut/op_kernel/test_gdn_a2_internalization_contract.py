"""A2/A3 与 A5 Phase6 内部化和架构隔离静态合同。"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
GDN_FWD = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd"
CORE = GDN_FWD / "chunk_gdn_core_fwd"
ARCH32 = CORE / "op_kernel/internal/arch32"
ARCH35 = CORE / "op_kernel/internal/arch35"
ENTRY = CORE / "op_kernel/chunk_gdn_core_fwd.cpp"
HOST_TILING = CORE / "op_host/chunk_gdn_core_fwd_tiling.cpp"
HOST_CMAKE = CORE / "op_host/CMakeLists.txt"
OP_DEF = CORE / "op_host/chunk_gdn_core_fwd_def.cpp"
OPTILING_CMAKE = ROOT / "cmake/obj_func.cmake"
ARCH32_ENTRY = ARCH32 / "chunk_gdn_core_fwd_arch32.cpp"
ARCH35_ENTRY = ARCH35 / "chunk_gdn_core_fwd_arch35.cpp"
ARCH32_STRUCT = ARCH32 / "chunk_gdn_core_fwd_arch32_struct.h"
ARCH35_STRUCT = ARCH35 / "chunk_gdn_core_fwd_arch35_struct.h"
ARCH32_TILING = (
    CORE / "op_host/op_tiling/arch32/chunk_gdn_core_fwd_arch32_tiling.cpp"
)
ARCH35_TILING = (
    CORE / "op_host/op_tiling/arch35/chunk_gdn_core_fwd_arch35_tiling.cpp"
)
ARCH32_STATE_TILING = (
    CORE
    / "op_host/op_tiling/arch32/"
    "chunk_gdn_core_fwd_arch32_state_output_tiling.cpp"
)

A2_KERNEL_SHA256 = {
    "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp":
        "6ed2af7e9c74375a41fcddcd04ac4de10ccf0d8d1e958afff5aca111e752bed5",
    "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp":
        "7d8fc24d7684e1f3ee0d707bef4f47498cd24563afe692b052ddc372630e6171",
    "chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp":
        "bf876192e45d90f901f99c9d0c541c3819c37525219fc7aec64d6c395ee065f8",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/"
        "block_epilogue_gdn_fwdh_vnew.hpp"
    ): "1a7a4c2fdd0b60bacc6e1d434cfbf5fc357cb71e6cb72096ef213d6abaaca6e1",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp"
    ): "b23a2086d88d3ae8333f7647cf1b7a053c8efd27f4304d34103bf5eb860e5032",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp"
    ): "1d401cd2048c6247277f1d458bcb59a13413b5c67651a2bc6f1a36d1f9ff9ed1",
    "solve_tri/op_kernel/solve_tri_cube.h":
        "5ceea9299ac0169692474f98d280b77173b16999af467a9055aa1edcbc3822e7",
}

A5_PUBLIC_SHA256 = {
    "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp":
        "6a930585c7504613b317e91f1ca4eafd83ba0d02e537124ee5835f1b36f4ad03",
    "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp":
        "b805fc7c3b42bf414c6fce2baf3222aae540a88209061878a993b52cfd19ba09",
    "chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp":
        "08064d1077aaa5f1a57d09acdef3a513a117b00afc027e384d471af9dbeab5b9",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/"
        "block_epilogue_gdn_fwdh_vnew.hpp"
    ): "3115ebf89f13f89aa7846d77e3708adcf1d3c9cabe33492554aa8f0175654208",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp"
    ): "0d9a2eceeda4460579692ee76e1d2a2152a157965d153e965e4ca6ee3a407d1a",
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/"
        "gdn_fwd_h_kernel.hpp"
    ): "8bd6bd8a55830e36969de5e31bbab08a7f05f32a200ad69fddf48028ef6113df",
    "solve_tri/op_kernel/solve_tri_cube.h":
        "a0d68575011a445784465a99c8f450ecd334239ad9bbed40c9b48f418f06e996",
}


def normalized_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def struct_field_types(text: str, name: str) -> tuple[str, ...]:
    body = text.split(f"struct {name} {{", 1)[1].split("};", 1)[0]
    return tuple(
        line.rstrip(";").rsplit(maxsplit=1)[0]
        for line in (item.strip() for item in body.splitlines())
        if line and not line.startswith("//")
    )


def test_public_entry_and_host_are_thin_architecture_dispatchers():
    entry = ENTRY.read_text(encoding="utf-8")
    host = HOST_TILING.read_text(encoding="utf-8")
    cmake = HOST_CMAKE.read_text(encoding="utf-8")

    assert entry.count("internal/arch32/chunk_gdn_core_fwd_arch32.cpp") == 1
    assert entry.count("internal/arch35/chunk_gdn_core_fwd_arch35.cpp") == 1
    assert "internal/arch32/operators/" not in entry
    assert "internal/operators/" not in entry
    assert "__CCE_AICORE__ == 310" in entry

    assert "Tiling4ChunkGdnCoreFwdArch32(context)" in host
    assert "Tiling4ChunkGdnCoreFwdArch35(context)" in host
    assert "SocVersion::ASCEND950" in host
    assert "internal/" not in host

    assert "internal/arch32/operators" in cmake
    assert "internal/coefficient_generation" in cmake
    assert "chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel" not in cmake
    assert "chunk_gdn_fwd/chunk_recompute_wu_fwd_ho" not in cmake


def test_architecture_entries_and_host_tiling_are_isolated():
    arch32_entry = ARCH32_ENTRY.read_text(encoding="utf-8")
    arch35_entry = ARCH35_ENTRY.read_text(encoding="utf-8")
    arch32_tiling = ARCH32_TILING.read_text(encoding="utf-8")
    arch35_tiling = ARCH35_TILING.read_text(encoding="utf-8")

    assert arch32_entry.count('#include "operators/') == 2
    assert "arch35" not in arch32_entry
    assert "Arch32ChunkGdnCoreFwdTrailer" in arch32_entry
    assert "Tiling4ChunkGdnCoreFwdArch32StateOutput" in arch32_tiling
    assert "arch35" not in arch32_tiling

    assert '#include "../coefficient_generation/' in arch35_entry
    assert "arch32" not in arch35_entry
    assert "Arch35ChunkGdnCoreFwdTrailer" in arch35_entry
    assert "Tiling4ChunkGdnCoreStateOutput" in arch35_tiling
    assert "arch32" not in arch35_tiling


def test_arch32_contains_no_arch35_implementation_or_reference():
    files = tuple(
        path
        for path in ARCH32.rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    assert files
    assert all("arch35" not in path.parts for path in files)
    assert all("arch35" not in path.read_text(encoding="utf-8") for path in files)


def test_frozen_a2_kernels_remain_private_and_a5_public_kernels_unchanged():
    for relative, expected in A2_KERNEL_SHA256.items():
        assert normalized_sha256(ARCH32 / "operators" / relative) == expected
    for relative, expected in A5_PUBLIC_SHA256.items():
        assert normalized_sha256(GDN_FWD / relative) == expected


def test_arch32_and_arch35_trailer_binary_layouts_match():
    arch32 = ARCH32_STRUCT.read_text(encoding="utf-8")
    arch35 = ARCH35_STRUCT.read_text(encoding="utf-8")

    assert struct_field_types(
        arch32, "Arch32ChunkGdnCoreFwdAbcTiling"
    ) == struct_field_types(
        arch35, "Arch35ChunkGdnCoreCoefficientTiling"
    )
    assert struct_field_types(
        arch32, "Arch32ChunkGdnCoreFwdTrailer"
    )[1:] == struct_field_types(
        arch35, "Arch35ChunkGdnCoreFwdTrailer"
    )[1:]


def test_arch32_state_tiling_keeps_frozen_a2_body():
    assert normalized_sha256(ARCH32_STATE_TILING) == (
        "f949736f99ced762db6b6276d3e6d46092ed8a48f0a410c4f8db4fe3501a9da5"
    )


def test_architecture_tiling_files_are_discovered_by_host_build():
    framework = OPTILING_CMAKE.read_text(encoding="utf-8")

    assert ARCH32_STATE_TILING.is_file()
    assert ARCH32_TILING.is_file()
    assert ARCH35_TILING.is_file()
    assert (
        "file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)"
        in framework
    )


def test_phase6_op_def_registers_a2_a3_and_a5():
    op_def = OP_DEF.read_text(encoding="utf-8")

    assert 'AddConfig("ascend910b", config)' in op_def
    assert 'AddConfig("ascend910_93", config)' in op_def
    assert 'AddConfig("ascend950", config)' in op_def
