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
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]          # torch_custom/fla_npu
CSRC_STABLE = ROOT / "csrc_stable" / "src"
CSRC_THIN = ROOT / "csrc_thin"
CSRC_STABLE_INCLUDE = ROOT / "csrc_stable" / "include"
GENERATED = ROOT / "csrc_stable" / "generated" / "ops_stable_generated.inc"


def generated_hash() -> str:
    """md5 of the generated adapters this build compiles in.

    The build stamp below and `_stable_generated._GENERATED_HASH` are the same
    value, and the Python side refuses to call into a library whose stamp does
    not match the glue it was imported with -- a .so left over from an earlier
    codegen run otherwise fails deep inside the dispatcher (or worse, silently
    launches on a stale stream).

    Newlines are normalised first: git checks the file out with CRLF on Windows
    and LF everywhere else, and a stamp that changes with the checkout would
    reject a perfectly good library.
    """

    return hashlib.md5(_normalised(GENERATED.read_bytes())).hexdigest()


def _normalised(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n")


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
    parser.add_argument("--no-debug-probe", action="store_true",
                        help="drop the _stream_probe op (smaller symbol surface)")
    args = parser.parse_args()

    includes, torch_lib = torch_paths()
    cmd = [
        args.compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        *(["-DFLA_STABLE_NO_DEBUG_PROBE"] if args.no_debug_probe else []),
        f'-DFLA_STABLE_SOURCE_HASH="{generated_hash()}"',
        "-I", str(CSRC_THIN / "include"),
        "-I", str(CSRC_STABLE_INCLUDE),
        *[f"-I{path}" for path in includes],
        # One TU by construction: the stable headers may not be included twice
        # (non-inline definitions in tensor_inl.h -> duplicate symbols).
        str(CSRC_STABLE / "stable_ops.cpp"),
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
