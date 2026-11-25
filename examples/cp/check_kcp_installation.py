#!/usr/bin/env python3
"""Check a filtered ten-operator KDA CP wheel without launching device work.

The full-project packaging check additionally requires unrelated FLA operators.
This scoped check does not relax that check or claim full-project acceptance.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    import fla_npu
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _stable, _stable_hash, _kcp
    from fla_npu._op_abi import elf_exported_symbols

    source = Path(__file__).resolve().parents[2] / "scripts/check_packaged_wheel_api.py"
    spec = importlib.util.spec_from_file_location("wheel_api_check", source)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    _stable.load()  # Also verifies the mandatory launcher build stamp.
    assert set(ascendc._ASCENDC_OPS) == {"npu_"+name for name in check.ASCENDC_NAMES}
    for name in ascendc._ASCENDC_OPS:
        _stable._op(name)
    for name in ("chunk_kda_cp_fwd", "chunk_kda_cp_bwd", "create_kda_cp_plan",
                 "KdaCpCache", "KdaCpPlan", "KdaCpResult"):
        assert hasattr(ascendc, name), name
    root = Path(fla_npu.__file__).resolve().parent
    directories = list((root/"opp/vendors").glob("*/op_impl/ai_core/tbe/kernel/config/*"))
    operators = ("kcp_all_gather_merge", "chunk_kda_fwd_prepare", "pre_process_fwd_kernel_merged",
        "merge_fwd_bwd_kernel", "chunk_fwd_h", "chunk_kda_fwd_finalize", "chunk_kda_bwd_prepare",
        "pre_process_bwd_kernel_merged", "chunk_gated_delta_rule_bwd_dhu", "chunk_kda_bwd_finalize")
    for name in operators:
        assert any((directory/(name+".json")).is_file() for directory in directories), name
    check._require_safe_packaged_opapi(fla_npu)
    check._require_packaged_triton_sources(fla_npu)
    _kcp._check_dependencies("save", "mc2", 2)
    library = Path(os.environ["FLA_NPU_OP_API_LIB"]).resolve()
    exports = elf_exported_symbols(library)
    assert {"aclnnKcpAllGatherMerge", "aclnnKcpAllGatherMergeGetWorkspaceSize"} <= exports
    assert not any(name.startswith("aclnnInnerKcpAllGatherMerge") for name in exports)
    assert not ascendc.FALLBACKS
    print(json.dumps(dict(status="passed", scope="KDA CP ten-operator wheel; no device launch",
        package=str(root), launcher_source_hash=_stable_hash.SOURCE_HASH,
        op_api_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        configurations=list(operators), registered_operators=len(ascendc._ASCENDC_OPS),
        full_project_packaging_accepted=False), indent=2))


if __name__ == "__main__":
    main()
