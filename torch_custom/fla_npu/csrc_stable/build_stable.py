#!/usr/bin/env python3
"""Build libfla_npu_thin.so against the torch Stable ABI.

Only `torch/csrc/stable/*` headers are on the include path; the produced shared
library must not reference any C++ ATen/c10 symbol (the ELF audit in
tools/stable_abi_audit.py asserts that).

Link line: the artifact is a *plain* shared library (no Python module), so it is
loaded later via torch.ops.load_library().  We link libtorch_cpu.so only so the
aoti_torch_* shims resolve; the loader reuses the already-loaded library through
its unversioned SONAME, which is what makes the artifact version-agnostic.

Usage:
  python csrc_stable/build_stable.py [--out PATH] [--torch-include DIR] ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]          # torch_custom/fla_npu
CSRC_STABLE = ROOT / "csrc_stable" / "src"
CSRC_THIN = ROOT / "csrc_thin"


def torch_paths() -> tuple[list[str], str]:
    import torch
    from torch.utils import cpp_extension

    includes = list(cpp_extension.include_paths())
    lib_torch = Path(torch.__file__).parent / "lib"
    return includes, str(lib_torch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "libfla_npu_thin.so"))
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    includes, torch_lib = torch_paths()
    cmd = [
        args.compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-I", str(CSRC_THIN / "include"),
        *[f"-I{path}" for path in includes],
        str(CSRC_STABLE / "stable_recurrent_gdr.cpp"),
        str(CSRC_THIN / "src" / "runtime.cpp"),
        "-L", torch_lib,
        "-Wl,--no-as-needed", "-ltorch_cpu", "-lc10", "-ltorch",
        "-ldl",
        "-o", args.out,
    ]
    print(" ".join(cmd))
    if shutil.which(args.compiler) is None:
        raise SystemExit(f"compiler {args.compiler!r} not found")
    subprocess.run(cmd, check=True)
    print(f"built {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
