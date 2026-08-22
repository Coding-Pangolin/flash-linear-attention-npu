#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Prepare a self-contained offline third-party bundle for embedding in the wheel.

The wheel ships a prebuilt OPP so ``pip install`` works offline for end users.
Developers who need to (re)build the wheel from the matching sources can extract
this offline bundle into the source tree's ``third_party/`` (CANN_3RD_LIB_PATH)
to compile fully offline, without reaching gitcode/gitee for the third-party
components.

The bundle keeps only the **minimal source subset** needed to (re)build:

- header-only libs (json / eigen / catlass) ship just their include trees;
- makeself ships its few files (probed by ``makeself.sh`` / ``makeself-header.sh``);
- abseil / protobuf are built by ``ExternalProject`` from the ``pkg/`` sources
  tarballs (not from a pre-laid source dir), so the bundle keeps only their
  ``pkg/*.tar.gz`` archives and leaves the CMake side to extract + patch them into
  the source tree at configure/build time.
- opbase ships its full source (it is compiled directly), with non-build dirs pruned.

Layout matches CANN_3RD_LIB_PATH detection:

    pkg/abseil-cpp-20230802.1.tar.gz   pkg/protobuf-25.1.tar.gz   pkg/include.zip
    pkg/eigen-5.0.0.tar.gz            pkg/googletest-1.14.0.tar.gz
    pkg/makeself-release-2.5.0-patch1.tar.gz
    json/include/    eigen/Eigen/    makeself/    opbase/    catlass/include/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# (name, source sub-path in cache, target sub-path inside bundle, prune list)
#   prune: top-level entries removed from the copied target dir to keep it minimal.
# abseil / protobuf sources are NOT copied here: ExternalProject extracts them from
#   pkg/*.tar.gz at build time, so only their archives are staged (see _ARCHIVES).
_COMPONENTS = [
    ("json", "json", "json", ["single_include"]),
    ("eigen", "eigen", "eigen", ["bench", "blas", "ci", "cmake", "debug", "demos", "doc", "failtest", "test"]),
    ("makeself", "makeself", "makeself", []),
    ("opbase", "opbase", "opbase", ["docs"]),
    ("catlass", "catlass/include", "catlass/include", []),
]

# name -> archive bytes to copy into bundle/pkg (source is cache root or cache/pkg)
_ARCHIVES = [
    "abseil-cpp-20230802.1.tar.gz",
    "protobuf-25.1.tar.gz",
    "include.zip",
    "eigen-5.0.0.tar.gz",
    "googletest-1.14.0.tar.gz",
    "makeself-release-2.5.0-patch1.tar.gz",
]


def _copy_dir(src: Path, dst_root: Path, target: str, prune: list[str]) -> None:
    target_dir = dst_root / target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target_dir)
    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)
    for name in prune:
        item = target_dir / name
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="third_party",
        help="source third-party cache directory (default: repository third_party/)",
    )
    parser.add_argument(
        "--out",
        default="offline_bundle",
        help="output bundle directory (default: offline_bundle/)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache = Path(args.cache)

    if not cache.is_absolute():
        cache = repo_root / cache
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out

    if not cache.is_dir():
        raise SystemExit(f"cache directory not found: {cache}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)

    for name, src_path, target, prune in _COMPONENTS:
        src = cache / src_path
        if not src.is_dir():
            print(f"[warn] missing '{name}' source at {src}; skipping", flush=True)
            continue
        _copy_dir(src, out, target, prune)
        print(f"[ok] {name}: {src} -> {out / target}", flush=True)

    for archive in _ARCHIVES:
        src_candidates = [cache / archive, cache / "pkg" / archive]
        src = next((c for c in src_candidates if c.is_file()), None)
        if src is None:
            print(f"[warn] missing archive '{archive}' in cache", flush=True)
            continue
        shutil.copy2(src, pkg / archive)
        print(f"[ok] archive {archive}", flush=True)

    print(f"\n[fla-npu] offline bundle ready: {out}")
    size_mb = sum(
        f.stat().st_size for f in out.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"[fla-npu] bundle size (uncompressed): {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
