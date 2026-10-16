#!/usr/bin/env python3
"""把 pre_process_fwd_kernel_merged 接入 fla_npu 的 Stable-ABI 适配层（幂等）。

做四件事（对应 docs/architecture/适配层接入指南.md §1 的交付件）：
  1. 新建 csrc/src/stable_pre_process_fwd_kernel_merged.cpp
  2. stable_ops.cpp：1 行 #include（按名字排序的位置）+ 2 行注册
     （m.def 插在 STABLE_TORCH_LIBRARY 块末尾、m.impl 插在
      STABLE_TORCH_LIBRARY_IMPL 块末尾，两个列表顺序保持一致）
  3. _stable.py：追加真签名 wrapper
  4. __init__.py：_ASCENDC_OPS 加一行 public 名

用法：
    python3 apply_stable_abi.py --repo /workspace/bartonfang/flash-linear-attention-npu
    python3 apply_stable_abi.py --repo <repo> --dry-run

脚本只做文本插入，幂等（已存在就跳过），不会替你跑门禁/编译。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OP = "pre_process_fwd_kernel_merged"
FULL = "npu_" + OP
SRC_FILE = f"stable_{OP}.cpp"
SCHEMA = f"kSchema_{OP}"
RUN = f"run_{FULL}"
MARK = f"# [stable-abi adapter] {FULL}"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str, dry: bool) -> None:
    if dry:
        print(f"    (dry-run) would write {path}")
        return
    path.write_text(text, encoding="utf-8")


def block_end(text: str, header: str) -> int:
    """返回 header 所在块 `{ ... }` 的右花括号下标。"""
    start = text.find(header)
    if start < 0:
        raise SystemExit(f"ERROR: 找不到 {header!r}")
    i = text.index("{", start)
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise SystemExit(f"ERROR: {header!r} 花括号不平衡")


def paren_end(text: str, header: str) -> int:
    """返回 header 所在 `( ... )` 块的右括号下标（用于 `_ASCENDC_OPS = (...)`）。"""
    start = text.find(header)
    if start < 0:
        raise SystemExit(f"ERROR: 找不到 {header!r}")
    i = text.index("(", start)
    depth = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise SystemExit(f"ERROR: {header!r} 括号不平衡")


def insert_before(text: str, pos: int, snippet: str) -> str:
    line_start = text.rfind("\n", 0, pos) + 1
    return text[:line_start] + snippet + text[line_start:]


def patch_ops_cpp(repo: Path, dry: bool) -> None:
    path = repo / "torch_custom/fla_npu/csrc/src/stable_ops.cpp"
    text = read(path)
    changed = []

    inc = f'#include "{SRC_FILE}"'
    if inc not in text:
        # 新版（一算子一文件）按名字排序：stable_pre_process... 排在
        # stable_kda_gate_cumsum.cpp 之后、stable_prepare_wy_repr_bwd.cpp 之前。
        # 旧版（拆分前，stable_chunk.cpp/stable_gdn.cpp 那种）没有同样锚点，
        # 退回"插在最后一行 #include \"stable_*.cpp\" 之后"。
        anchor = '#include "stable_prepare_wy_repr_bwd.cpp"'
        pos = text.find(anchor)
        if pos >= 0:
            text = insert_before(text, pos, inc + "\n")
        else:
            lines = text.splitlines(keepends=True)
            last = -1
            for i, line in enumerate(lines):
                if line.startswith('#include "stable_') and line.rstrip().endswith('.cpp"'):
                    last = i
            if last < 0:
                raise SystemExit("ERROR: stable_ops.cpp 里找不到 include 锚点")
            lines.insert(last + 1, inc + "\n")
            text = "".join(lines)
        changed.append("include")
    else:
        print("    include 已存在，跳过")

    if f"m.def({SCHEMA})" not in text:
        probe = "#ifndef FLA_STABLE_NO_DEBUG_PROBE"
        pos = text.find(probe)
        if pos < 0:
            pos = block_end(text, "STABLE_TORCH_LIBRARY(fla_npu_stable, m)")
        text = insert_before(text, pos, f"  m.def({SCHEMA});\n")
        changed.append("m.def")
    else:
        print("    m.def 已存在，跳过")

    if f'm.impl("{FULL}"' not in text:
        # def / impl 两个列表要按位置配对；_stream_probe 在两个列表里都在末尾，
        # 所以我们也插在各自列表的 _stream_probe 之前（找不到就插块末尾）。
        probe = "#ifndef FLA_STABLE_NO_DEBUG_PROBE"
        first = text.find(probe)
        second = text.find(probe, first + 1) if first >= 0 else -1
        pos = second if second >= 0 else block_end(text, "STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable")
        snippet = (
            f'  m.impl("{FULL}",\n'
            f"         &fla_npu_stable::stable::boxed_adapter<{RUN}>);\n"
        )
        text = insert_before(text, pos, snippet)
        changed.append("m.impl")
    else:
        print("    m.impl 已存在，跳过")

    write(path, text, dry)
    print(f"    stable_ops.cpp: {', '.join(changed) if changed else '无改动'}")


def patch_stable_py(repo: Path, dry: bool, wrapper: str) -> None:
    path = repo / "torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py"
    text = read(path)
    if MARK in text:
        print("    _stable.py: wrapper 已存在，跳过")
        return
    if not text.endswith("\n"):
        text += "\n"
    write(path, text + "\n" + MARK + "\n" + wrapper.rstrip("\n") + "\n", dry)
    print("    _stable.py: 追加 wrapper")


def patch_init_py(repo: Path, dry: bool) -> None:
    path = repo / "torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py"
    text = read(path)
    if f'"{FULL}"' in text:
        print("    __init__.py: 已存在，跳过")
        return
    pos = paren_end(text, "_ASCENDC_OPS = (")
    snippet = f'    "{FULL}",\n'
    text = insert_before(text, pos, snippet)
    write(path, text, dry)
    print("    __init__.py: _ASCENDC_OPS +1")


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="fla-npu 仓根目录（含 torch_custom/）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "torch_custom/fla_npu/csrc/src/stable_ops.cpp").is_file():
        raise SystemExit(f"ERROR: {repo} 不像 fla-npu 仓（缺 stable_ops.cpp）")

    src = here / SRC_FILE
    if not src.is_file():
        raise SystemExit(f"ERROR: 缺少 {src}")

    dst = repo / "torch_custom/fla_npu/csrc/src" / SRC_FILE
    print(f"[1/4] {dst}")
    if dst.exists():
        print("    已存在，覆盖")
    if not args.dry_run:
        shutil.copyfile(src, dst)

    print("[2/4] csrc/src/stable_ops.cpp")
    patch_ops_cpp(repo, args.dry_run)

    print("[3/4] fla_npu/ops/ascendc/_stable.py")
    wrapper = (here / "_stable_wrapper_snippet.py").read_text(encoding="utf-8")
    body = wrapper.split("# -----", 1)
    # 只取第一段（wrapper 本体），后面的说明性注释单独处理
    start = wrapper.find("def npu_pre_process_fwd_kernel_merged")
    end = wrapper.find("# ---------------------------------------------------------------------------\n# fla_npu/ops/ascendc/__init__.py")
    if start < 0 or end < 0:
        raise SystemExit("ERROR: wrapper snippet 解析失败")
    patch_stable_py(repo, args.dry_run, wrapper[start:end])

    print("[4/4] fla_npu/ops/ascendc/__init__.py")
    patch_init_py(repo, args.dry_run)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
