#!/usr/bin/env python3
"""Audit a Stable-ABI thin launcher: sources and ELF must stay on the C shims.

Two independent checks:

1. Source level — the stable translation units may only include
   ``torch/csrc/stable/*`` (plus vendored torch-free helpers); any ATen / c10 /
   pybind11 / torch/extension.h include re-introduces the unstable ABI.
2. ELF level — the produced shared object must not leave any C++ ATen/c10 symbol
   undefined.  Those mangled names (`_ZN2at*`, `_ZN3c10*`) are exactly what
   breaks when torch is upgraded while the shim symbols (`aoti_torch_*`) keep
   their ABI promise.

Usage:
  python stable_abi_audit.py --source ../csrc_stable/src/*.cpp
  python stable_abi_audit.py --lib ../libfla_npu_thin.so
  python stable_abi_audit.py --source <glob> --lib <path>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN_INCLUDES = (
    "ATen/",
    "c10/",
    "torch/extension.h",
    "torch/csrc/utils/",
    "torch/csrc/autograd/",
    "torch/csrc/api/",
    "pybind11/",
)
ALLOWED_PREFIXES = ("torch/csrc/stable/", "torch/headeronly/")

MANGLED_UNSTABLE = (re.compile(r"^_ZN2at"), re.compile(r"^_ZN3c10"),
                    re.compile(r"^_ZN8pybind11"))

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


def audit_source(path: Path) -> list[str]:
    problems = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for include in _INCLUDE_RE.findall(text):
        if include.startswith(ALLOWED_PREFIXES):
            continue
        if any(bad in include for bad in FORBIDDEN_INCLUDES):
            problems.append(f"{path.name}: forbidden include <{include}>")
    return problems


def audit_lib(path: Path) -> list[str]:
    problems = []
    out = subprocess.run(
        ["nm", "-D", "--undefined-only", str(path)],
        check=True, capture_output=True, text=True).stdout
    undefined = [line.split()[-1] for line in out.splitlines() if line.strip()]
    unstable = [name for name in undefined
                if any(pattern.match(name) for pattern in MANGLED_UNSTABLE)]
    if unstable:
        problems.append(
            f"{path.name}: {len(unstable)} unstable C++ symbol(s) referenced, "
            f"e.g. {unstable[:3]}")
    needs = subprocess.run(["readelf", "-d", str(path)], check=True,
                           capture_output=True, text=True).stdout
    if "libtorch_python.so" in needs:
        problems.append(f"{path.name}: links libtorch_python.so (Python ABI)")
    shims = [name for name in undefined if name.startswith("aoti_torch_")]
    print(f"  ELF: {len(undefined)} undefined symbols, {len(shims)} aoti_torch_*")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="*", type=Path, default=[])
    parser.add_argument("--lib", type=Path)
    args = parser.parse_args()

    problems: list[str] = []
    for source in args.source:
        problems.extend(audit_source(source))
    if args.lib:
        problems.extend(audit_lib(args.lib))
    if problems:
        print("FAIL stable-abi audit")
        for line in problems:
            print("  -", line)
        return 1
    print("OK stable-abi audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
