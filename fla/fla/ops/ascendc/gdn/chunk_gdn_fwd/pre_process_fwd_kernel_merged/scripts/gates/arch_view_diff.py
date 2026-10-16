"""按 arch 分档展开条件编译，比较"某一平台视角下真正参与编译的源码"。

用途：本算子用 `#if PPFM_ARCH_IS_950 / #else / #endif`（以及 PPFM_* 开关）分档，
改了 910B 分支后需要证明"950 视角一行没动"，或在 246 网络不通、不能重建时给出等价性证据。

用法：
    # 取某个提交的版本
    git show <rev>:<path> > /tmp/v1.cpp
    python3 arch_view_diff.py --macros "__CCE_AICORE__=310,PPFM_TILE_MMAD=1" v1.cpp v2.cpp

只支持该文件用到的表达式形式：defined(X)/!defined(X)/整数比较/&&/||/!/括号，
以及 `#define NAME <整数>` 的简单宏。
"""
from __future__ import annotations

import argparse
import re
import sys


class Expander:
    def __init__(self, macros: dict[str, int]) -> None:
        self.macros = dict(macros)

    def eval_expr(self, expr: str) -> bool:
        e = expr
        e = re.sub(r"defined\s*\(\s*(\w+)\s*\)",
                   lambda m: "1" if m.group(1) in self.macros else "0", e)
        e = re.sub(r"defined\s+(\w+)",
                   lambda m: "1" if m.group(1) in self.macros else "0", e)
        e = e.replace("!=", " NE ")
        e = re.sub(r"\b([A-Za-z_]\w*)\b",
                   lambda m: str(self.macros.get(m.group(1), 0)), e)
        e = e.replace("&&", " and ").replace("||", " or ")
        e = e.replace("!", " not ")
        e = e.replace("NE", "!=")
        try:
            return bool(eval(e, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as exc:  # pragma: no cover - 诊断用
            raise SystemExit(f"无法求值条件 '{expr}' -> '{e}': {exc}")


def active_view(path: str, macros: dict[str, int]) -> list[str]:
    exp = Expander(macros)
    stack: list[tuple[bool, bool]] = []  # (父有效, 本分支有效)
    out: list[str] = []
    for raw in open(path, encoding="utf-8").read().splitlines():
        line = raw.strip()
        if line.startswith("#if ") or line.startswith("#elif "):
            parent = stack[-1][0] if stack else True
            cond = exp.eval_expr(line.split(" ", 1)[1])
            stack.append((parent, parent and cond))
            continue
        if line.startswith("#ifdef ") or line.startswith("#ifndef "):
            parent = stack[-1][0] if stack else True
            name = line.split(" ", 1)[1].strip()
            present = name in exp.macros
            cond = present if line.startswith("#ifdef") else not present
            stack.append((parent, parent and cond))
            continue
        if line.startswith("#else"):
            parent = stack[-1][0] if stack else True
            was = stack[-1][1]
            stack[-1] = (parent, parent and not was)
            continue
        if line.startswith("#endif"):
            if stack:
                stack.pop()
            continue
        if not all(entry[1] for entry in stack):
            continue
        m = re.match(r"#define\s+(\w+)\s+(-?\d+)\s*$", line)
        if m:
            exp.macros[m.group(1)] = int(m.group(2))
        out.append(raw.rstrip("\n"))
    return out


def strip_comments(lines: list[str]) -> list[str]:
    """去掉行尾 `//` 注释与纯空行 —— 只比较真正参与编译的代码文本。"""
    out: list[str] = []
    for line in lines:
        body = line.split("//", 1)[0].rstrip()
        if body:
            out.append(body)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--macros", default="")
    ap.add_argument("--ignore-comments", action="store_true",
                    help="忽略注释/空行，只比较代码（推荐）")
    args = ap.parse_args()

    macros: dict[str, int] = {}
    for kv in filter(None, (s.strip() for s in args.macros.split(","))):
        k, _, v = kv.partition("=")
        macros[k.strip()] = int(v or "1")

    a = active_view(args.old, macros)
    b = active_view(args.new, macros)
    if args.ignore_comments:
        a, b = strip_comments(a), strip_comments(b)
    print(f"[arch-view] macros={macros}")
    print(f"[arch-view] old 活跃行={len(a)}  new 活跃行={len(b)}")

    import difflib
    diff = list(difflib.unified_diff(a, b, fromfile=args.old, tofile=args.new, lineterm=""))
    if not diff:
        print("[arch-view] 该平台视角下源码完全一致 ✅")
        return 0
    print(f"[arch-view] 差异 {len(diff)} 行：")
    for line in diff[:200]:
        print(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
