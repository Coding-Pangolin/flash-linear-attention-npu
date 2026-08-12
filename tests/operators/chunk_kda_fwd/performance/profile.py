#!/usr/bin/env python3
"""Profile JSON performance cases for chunk_kda_fwd with msopprof."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.chunk_kda_fwd.common.case_matrix import case_ids


RUNNER = ROOT / "tests/operators/_shared/chunk_kda_backend.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--case-id",
        default="chunk_kda_fwd_performance",
        help="exact performance case id from tests/op_cases/chunk_kda_fwd.json",
    )
    parser.add_argument("--output", default="outputs/chunk_kda_fwd_msopprof")
    parser.add_argument("--launch-count", type=int, default=20)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--kernel-name",
        help="optional exact or profiler-supported kernel name filter",
    )
    parser.add_argument(
        "--aic-metrics",
        default="BasicInfo",
        help="comma-separated msopprof metric groups",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("application", "kernel", "range"),
        default="application",
    )
    parser.add_argument("--kill", choices=("on", "off"), default="on")
    args = parser.parse_args()
    ids = case_ids(tag="performance", route="ascendc")
    if args.case_id not in ids:
        raise RuntimeError(
            f"unknown chunk_kda_fwd performance case {args.case_id!r}; "
            f"available cases: {', '.join(ids)}"
        )
    env = os.environ.copy()
    env.update({
        "FLA_NPU_OPERATOR": "chunk_kda_fwd",
        "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/chunk_kda_fwd.json"),
        "FLA_NPU_CASE_IDS": args.case_id,
        "FLA_NPU_PROFILE_ONLY": "1",
        "FLA_NPU_PROFILE_REPEATS": str(args.repeats),
    })
    application = f"{shlex.quote(sys.executable)} {shlex.quote(str(RUNNER))}"
    command = [
        "msopprof", f"--application={application}", f"--output={args.output}",
        f"--aic-metrics={args.aic_metrics}", f"--launch-count={args.launch_count}",
        f"--warm-up={args.warm_up}", f"--replay-mode={args.replay_mode}",
        f"--kill={args.kill}",
    ]
    if args.kernel_name:
        command.append(f"--kernel-name={args.kernel_name}")
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return
    subprocess.run(command, cwd=RUNNER.parent, env=env, check=True)


if __name__ == "__main__":
    main()
