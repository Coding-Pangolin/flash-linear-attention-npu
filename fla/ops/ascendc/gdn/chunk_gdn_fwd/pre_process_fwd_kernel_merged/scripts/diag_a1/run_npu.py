#!/usr/bin/env python3
"""Preload torch_npu then run a target script (241 版启动器).

usage: python run_npu.py <target.py> [args...]
"""

from __future__ import annotations

import runpy
import sys

import torch_npu  # noqa: F401  (side effect: registers the npu device backend)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
