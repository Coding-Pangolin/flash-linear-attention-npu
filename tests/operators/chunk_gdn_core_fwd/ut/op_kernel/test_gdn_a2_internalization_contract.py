"""A2/A3 Phase6 内部化与架构隔离静态合同。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
GDN_FWD = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd"
CORE = GDN_FWD / "chunk_gdn_core_fwd"
ARCH32 = CORE / "op_kernel/internal/arch32"
ENTRY = CORE / "op_kernel/chunk_gdn_core_fwd.cpp"
HOST_TILING = CORE / "op_host/chunk_gdn_core_fwd_tiling.cpp"
HOST_CMAKE = CORE / "op_host/CMakeLists.txt"
ARCH32_STATE_TILING = (
    CORE / "op_host/arch32/chunk_gdn_core_fwd_arch32_state_output_tiling.cpp"
)


COPIED_KERNEL_PAIRS = (
    (
        "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp",
        "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp",
    ),
    (
        "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp",
        "chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp",
    ),
    (
        "chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp",
        "chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp",
    ),
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/"
        "block_epilogue_gdn_fwdh_vnew.hpp",
        "chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/"
        "block_epilogue_gdn_fwdh_vnew.hpp",
    ),
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp",
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/block/"
        "block_scheduler_gdn_fwd_h.hpp",
    ),
    (
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp",
        "chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp",
    ),
    (
        "solve_tri/op_kernel/solve_tri_cube.h",
        "solve_tri/op_kernel/solve_tri_cube.h",
    ),
)


def internal_operator(path: str) -> Path:
    return ARCH32 / "operators" / path


def test_phase6_entry_and_host_only_reference_arch32_private_sources():
    entry = ENTRY.read_text(encoding="utf-8")
    host = HOST_TILING.read_text(encoding="utf-8")
    cmake = HOST_CMAKE.read_text(encoding="utf-8")

    assert entry.count("internal/arch32/operators/") == 2
    assert "../../chunk_recompute_wu_fwd_ho" not in entry
    assert "../../chunk_kkt_solve_tri" not in entry
    assert "Tiling4ChunkGdnCoreFwdArch32StateOutput" in host
    assert "Tiling4ChunkRecomputeWUFwdHO(context)" not in host
    assert "internal/arch32/operators" in cmake
    assert "chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel" not in cmake
    assert "chunk_gdn_fwd/chunk_recompute_wu_fwd_ho" not in cmake


def test_arch32_contains_no_arch35_implementation_or_reference():
    assert ARCH32.is_dir()
    files = tuple(
        path
        for path in ARCH32.rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    assert files
    assert all("arch35" not in path.parts for path in files)
    assert all("arch35" not in path.read_text(encoding="utf-8") for path in files)


def test_accuracy_sensitive_arch32_files_match_frozen_a2_sources_byte_for_byte():
    for source_name, internal_name in COPIED_KERNEL_PAIRS:
        source = GDN_FWD / source_name
        internal = internal_operator(internal_name)
        assert source.is_file(), source
        assert internal.is_file(), internal
        assert internal.read_bytes() == source.read_bytes(), source_name


def test_arch32_state_tiling_keeps_frozen_a2_body():
    source = (
        GDN_FWD
        / "chunk_recompute_wu_fwd_ho/op_host/"
        "chunk_recompute_wu_fwd_ho_tiling.cpp"
    ).read_text(encoding="utf-8")
    internal = ARCH32_STATE_TILING.read_text(encoding="utf-8")

    source_body = source.split(
        "ge::graphStatus Tiling4ChunkRecomputeWUFwdHO", 1
    )[1].split("\nge::graphStatus TilingPrepareForChunkRecomputeWUFwdHO", 1)[0]
    internal_body = internal.split(
        "ge::graphStatus Tiling4ChunkGdnCoreFwdArch32StateOutput", 1
    )[1].split("\n} // namespace optiling", 1)[0]
    internal_body = internal_body.replace(
        "Tiling4ChunkGdnCoreFwdArch32StateOutput start.",
        "Tiling4ChunkRecomputeWUFwdHO start.",
    )
    assert internal_body == source_body
