# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify that the running fla_npu really loads freshly built artifacts.

Two kinds of artifacts are compared between what ``import fla_npu`` actually
loads at runtime and the freshly built/staged copies:

1. OPP library: ``libcust_opapi.so``
   - runtime: from ``FLA_NPU_OP_API_LIB``, set during ``import fla_npu``;
   - built:   ``build/libcust_opapi.so`` by default, or ``--built-lib`` /
     ``--run-package`` (extracted from a Makeself run package).
2. Python wrapper: the core ``.py`` files under the installed package
   (``fla_npu/__init__.py``, ``ops/ascendc/*.py``), compared against the
   sources under ``torch_custom/fla_npu/fla_npu/``.

If both sides are found, it reports whether their md5 match.  ``[OK]`` means
the running wheel is exactly the one freshly built; ``[FAIL]`` means a stale
copy is still in use and the new wheel / run package must be reinstalled.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


# Python wrapper files that are pure copies of the sources (no build-time
# rewriting), as paths relative to the fla_npu package root, so the runtime
# side can be located through the installed package directory.
PYTHON_WRAPPER_MODULES = (
    "__init__.py",
    "ops/ascendc/__init__.py",
    "ops/ascendc/_aclnn_ctypes.py",
    "ops/ascendc/_runtime.py",
)


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


def _check_opp_lib(repo_root: Path, built_lib: str | None, run_package: str | None) -> bool:
    loaded_env = os.environ.get("FLA_NPU_OP_API_LIB")
    loaded_lib = Path(loaded_env) if loaded_env else None
    if loaded_lib is None or not loaded_lib.exists():
        print(f"[FAIL] runtime libcust_opapi.so not found (FLA_NPU_OP_API_LIB={loaded_env!r})")
        return False

    loaded_md5 = _md5(loaded_lib)
    print(f"loaded libcust_opapi.so: {loaded_lib}")
    print(f"loaded md5:              {loaded_md5}")

    built_lib_path = _find_built_lib(repo_root, built_lib, run_package)
    if built_lib_path is None:
        print(
            "[WARN] no freshly built libcust_opapi.so found; pass --built-lib or --run-package "
            "to compare against a build artifact."
        )
        return True

    built_md5 = _md5(built_lib_path)
    print(f"built  libcust_opapi.so: {built_lib_path}")
    print(f"built  md5:              {built_md5}")

    if loaded_md5 == built_md5:
        print("[OK] loaded libcust_opapi.so matches the freshly built one.")
        return True
    print(
        "[FAIL] loaded libcust_opapi.so differs from the freshly built one; "
        "the running wheel still uses an older OPP. Reinstall the new wheel/run package."
    )
    return False


def _check_python_wrapper(repo_root: Path) -> bool:
    """Compare installed wrapper .py files against their sources."""
    import fla_npu

    installed_root = Path(fla_npu.__file__).resolve().parent
    source_root = repo_root / "torch_custom" / "fla_npu" / "fla_npu"
    if not source_root.exists():
        print(f"[WARN] wrapper source dir not found: {source_root}", file=sys.stderr)
        return True

    ok = True
    checked = 0
    for rel in PYTHON_WRAPPER_MODULES:
        installed = installed_root / rel
        source = source_root / rel
        if not installed.exists():
            print(f"[FAIL] installed wrapper file not found: {installed}")
            ok = False
            continue
        if not source.exists():
            print(f"[WARN] wrapper source file not found: {source}", file=sys.stderr)
            continue
        imd5 = _md5(installed)
        smd5 = _md5(source)
        status = "OK" if imd5 == smd5 else "FAIL"
        if imd5 != smd5:
            ok = False
        print(f"[{status}] {rel}")
        print(f"  loaded: {installed}  md5={imd5}")
        print(f"  source: {source}      md5={smd5}")
        checked += 1

    if checked == 0:
        print("[WARN] no Python wrapper files compared", file=sys.stderr)
        return True
    if ok:
        print("[OK] installed Python wrapper files match the current sources.")
    else:
        print(
            "[FAIL] some installed Python wrapper files differ from the current sources; "
            "reinstall the newly built wheel."
        )
    return ok


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
    parser.add_argument(
        "--python",
        action="store_true",
        help="Also compare the installed Python wrapper files against the current sources.",
    )
    args = parser.parse_args()

    # Import first so OPP env vars and module paths are set up.
    import fla_npu  # noqa: F401

    repo_root = Path(__file__).resolve().parents[1]

    results = [_check_opp_lib(repo_root, args.built_lib, args.run_package)]
    if args.python:
        results.append(_check_python_wrapper(repo_root))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
