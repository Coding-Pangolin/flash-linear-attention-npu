#!/usr/bin/env python3
"""apply_stable_abi.py 的自测：在临时目录里搭一个假仓，跑一遍并断言插入结果。

    python selftest_apply.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

# 假仓按 246 的"拆分前"布局：分组文件 + 两个 _stream_probe 块（def / impl 各一）
FAKE_OPS = '''#include "stable_runtime_common.cpp"

#include "stable_recurrent_gdr.cpp"
#include "stable_recurrent_kda.cpp"
#include "stable_kda.cpp"
#include "stable_chunk.cpp"
#include "stable_gdn.cpp"
#include "stable_fwd_h.cpp"

STABLE_TORCH_LIBRARY(fla_npu_stable, m) {
  m.def(kSchema_kda_gate_cumsum);
  m.def(kSchema_prepare_wy_repr_bwd);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.def("_stream_probe(int device_index) -> (int, int)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) {
  m.impl("npu_kda_gate_cumsum",
         &fla_npu_stable::stable::boxed_adapter<run_npu_kda_gate_cumsum>);
  m.impl("npu_prepare_wy_repr_bwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_prepare_wy_repr_bwd>);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.impl("_stream_probe", &boxed_stream_probe);
#endif
}
'''


def build_fake_repo(root: pathlib.Path) -> None:
    (root / "torch_custom/fla_npu/csrc/src").mkdir(parents=True)
    (root / "torch_custom/fla_npu/fla_npu/ops/ascendc").mkdir(parents=True)
    (root / "torch_custom/fla_npu/csrc/src/stable_ops.cpp").write_text(FAKE_OPS, encoding="utf-8")
    (root / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py").write_text(
        "# fake\n\n\ndef _op(name):\n    return name\n", encoding="utf-8")
    (root / "torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py").write_text(
        '_ASCENDC_OPS = (\n    "npu_kda_gate_cumsum",\n)\n', encoding="utf-8")


def check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    root = pathlib.Path(tempfile.mkdtemp(prefix="fakefla_"))
    failures: list[str] = []
    try:
        build_fake_repo(root)
        r = subprocess.run([sys.executable, str(HERE / "apply_stable_abi.py"),
                            "--repo", str(root)], capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr)
            return 1

        ops = (root / "torch_custom/fla_npu/csrc/src/stable_ops.cpp").read_text(encoding="utf-8")
        init = (root / "torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py").read_text(encoding="utf-8")
        stable = (root / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py").read_text(encoding="utf-8")
        opfile = root / "torch_custom/fla_npu/csrc/src/stable_pre_process_fwd_kernel_merged.cpp"

        check(opfile.is_file(), "算子文件已 copy", failures)
        check('#include "stable_pre_process_fwd_kernel_merged.cpp"' in ops, "include 已插入", failures)
        check(ops.index('#include "stable_pre_process_fwd_kernel_merged.cpp"')
              > ops.index('#include "stable_fwd_h.cpp"'), "include 插在最后一个 stable_ include 之后", failures)
        check("m.def(kSchema_pre_process_fwd_kernel_merged);" in ops, "m.def 已插入", failures)
        i_def = ops.index("m.def(kSchema_pre_process_fwd_kernel_merged);")
        i_probe1 = ops.index("#ifndef FLA_STABLE_NO_DEBUG_PROBE")
        check(i_def < i_probe1, "m.def 在 def 列表里排在 _stream_probe 之前", failures)
        check('m.impl("npu_pre_process_fwd_kernel_merged",' in ops, "m.impl 已插入", failures)
        i_impl = ops.index('m.impl("npu_pre_process_fwd_kernel_merged"')
        i_probe2 = ops.index("#ifndef FLA_STABLE_NO_DEBUG_PROBE", i_probe1 + 1)
        check(i_probe1 < i_impl < i_probe2, "m.impl 在 impl 列表里排在 _stream_probe 之前（与 def 位置对齐）", failures)
        check('"npu_pre_process_fwd_kernel_merged",' in init, "__init__._ASCENDC_OPS +1", failures)
        check("# [stable-abi adapter] npu_pre_process_fwd_kernel_merged" in stable, "_stable.py wrapper 已追加", failures)
        check("def npu_pre_process_fwd_kernel_merged(" in stable, "wrapper 函数名正确", failures)
        check(stable.count("_current_stream_ptr(),") >= 1, "wrapper 末尾取 stream", failures)

        # 幂等：再跑一次不应改变内容
        subprocess.run([sys.executable, str(HERE / "apply_stable_abi.py"), "--repo", str(root)],
                       capture_output=True, text=True)
        ops2 = (root / "torch_custom/fla_npu/csrc/src/stable_ops.cpp").read_text(encoding="utf-8")
        check(ops2 == ops, "重复 apply 幂等", failures)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("SELFTEST PASS" if not failures else f"SELFTEST FAIL: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
