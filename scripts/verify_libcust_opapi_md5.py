# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare the md5 of the loaded libcust_opapi.so with a freshly built one.

When a developer rebuilds the operators, they want to make sure the runtime
really loads the newly compiled ``libcust_opapi.so`` rather than a stale copy.
This script prints the md5 of:

1. the ``libcust_opapi.so`` that ``import fla_npu`` actually loads
   (from ``FLA_NPU_OP_API_LIB``, set during import);
2. a candidate build artifact, by default the one staged under ``build/``,
   or an explicit path passed with ``--built-lib`` / ``--run-package``.

If both are found, it reports whether they match.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_run_package_lib(run_package: Path) -> Path | None:
    """Locate libcust_opapi.so inside a .run package (Makeself self-extracting archive)."""
    import subprocess
    import tempfile

    if not run_package.exists():
        print(f"[WARN] run package not found: {run_package}", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory(prefix="fla-npu-run-extract-") as tmp:
        try:
            proc = subprocess.run(
                [str(run_package), f"--extract={tmp}", "--noexec"],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[WARN] failed to extract run package {run_package} "
                f"(exit {exc.returncode}):\n{exc.stderr.decode(errors='replace')[-500:]}",
                file=sys.stderr,
            )
            return None
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"[WARN] failed to extract run package {run_package}: {exc}", file=sys.stderr)
            return None
        extracted = Path(tmp) / "packages" / "vendors" / "fla_npu_transformer" / "op_api" / "lib" / "libcust_opapi.so"
        if not extracted.exists():
            return None
        # Copy to a stable location so the md5 survives tempdir cleanup.
        stable = run_package.parent / f".{run_package.name}.libcust_opapi.so"
        try:
            import shutil

            shutil.copy2(extracted, stable)
            return stable
        except OSError:
            return extracted


def _find_built_lib(repo_root: Path, built_lib: str | None, run_package: str | None) -> Path | None:
    if run_package:
        return _resolve_run_package_lib(Path(run_package))
    if built_lib:
        candidate = Path(built_lib)
        if candidate.exists():
            return candidate
        return None

    candidates = [
        repo_root / "build" / "libcust_opapi.so",
        repo_root / "build" / "lib" / "fla_npu" / "opp" / "vendors" / "fla_npu_transformer" / "op_api" / "lib" / "libcust_opapi.so",
    ]
    return next((path for path in candidates if path.exists()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--built-lib",
        default=None,
        help="Path to the freshly built libcust_opapi.so. Defaults to the one staged under build/.",
    )
    parser.add_argument(
        "--run-package",
        default=None,
        help="Path to a freshly built fla-npu-*.run package; its libcust_opapi.so is extracted for comparison.",
    )
    args = parser.parse_args()

    import fla_npu  # noqa: F401  (import sets FLA_NPU_OP_API_LIB)

    loaded_env = os.environ.get("FLA_NPU_OP_API_LIB")
    loaded_lib = Path(loaded_env) if loaded_env else None
    if loaded_lib is None or not loaded_lib.exists():
        print(f"[FAIL] runtime libcust_opapi.so not found (FLA_NPU_OP_API_LIB={loaded_env!r})")
        return 1

    loaded_md5 = _md5(loaded_lib)
    print(f"loaded libcust_opapi.so: {loaded_lib}")
    print(f"loaded md5:              {loaded_md5}")

    repo_root = Path(__file__).resolve().parents[1]
    built_lib = _find_built_lib(repo_root, args.built_lib, args.run_package)
    if built_lib is None:
        print(
            "[WARN] no freshly built libcust_opapi.so found; pass --built-lib or --run-package "
            "to compare against a build artifact."
        )
        return 0

    built_md5 = _md5(built_lib)
    print(f"built  libcust_opapi.so: {built_lib}")
    print(f"built  md5:              {built_md5}")

    if loaded_md5 == built_md5:
        print("[OK] loaded libcust_opapi.so matches the freshly built one.")
        return 0
    print(
        "[FAIL] loaded libcust_opapi.so differs from the freshly built one; "
        "the running wheel still uses an older OPP. Reinstall the new wheel/run package."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
