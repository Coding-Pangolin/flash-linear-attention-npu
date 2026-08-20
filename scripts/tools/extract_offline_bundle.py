#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Extract the wheel's embedded offline third-party bundle into a source tree.

The wheel ships two things: a prebuilt OPP (for offline use) and an embedded
offline third-party bundle under ``fla_npu/offline/third_party``.  This script
copies that bundle into a source tree's ``third_party/`` (CANN_3RD_LIB_PATH), so
the same machine can then (re)build the wheel/project fully offline without
reaching gitcode/gitee for the third-party components.

Usage:

    python scripts/tools/extract_offline_bundle.py --src <repo/clone-path>

The bundle layout matches exactly what ``cmake/third_party/*.cmake`` probes for
offline (abseil-cpp/, protobuf/, json/, eigen/, makeself/, opbase/,
catlass/include/, pkg/*.tar.gz), so a configure after extraction stays offline.

Safety: ``--src`` must point at a source tree (a directory containing
``CMakeLists.txt`` and ``build.sh``).  Existing third-party content under that
tree's ``third_party/`` is intentionally overwritten so every component matches
the wheel's pinned versions; mixing locally cached and wheel versions can break
the build.  The script never touches system paths or the installed package.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _installed_bundle() -> Path | None:
    """Locate the offline bundle inside the installed fla_npu package."""
    try:
        import fla_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] unable to import fla_npu: {exc}", flush=True)
        return None
    pkg_dir = Path(fla_npu.__file__).resolve().parent  # type: ignore[attr-defined]
    bundle = pkg_dir / "offline" / "third_party"
    if not bundle.is_dir():
        print(f"[warn] no embedded offline bundle at {bundle}", flush=True)
        return None
    return bundle


def _looks_like_source_tree(root: Path) -> bool:
    return (root / "CMakeLists.txt").is_file() and (root / "build.sh").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="path to the source tree whose third_party/ should be (over)written",
    )
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    if not src_root.is_dir():
        raise SystemExit(f"source tree not found: {src_root}")
    if not _looks_like_source_tree(src_root):
        raise SystemExit(
            f"{src_root} does not look like a flash-linear-attention-npu source "
            "tree (missing CMakeLists.txt and/or build.sh). Refusing to write."
        )

    bundle = _installed_bundle()
    if bundle is None:
        raise SystemExit(
            "could not locate the embedded offline bundle; is fla-npu installed "
            "with the offline bundle (see prepare_offline_bundle.py)?"
        )

    dest = src_root / "third_party"
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in bundle.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied += 1
        print(f"[ok] {item.name}", flush=True)

    print(
        f"\n[fla-npu] extracted and overwrote {copied} components under {dest} "
        "(matched wheel-pinned versions).",
        flush=True,
    )
    print(
        "[fla-npu] source-tree third_party is ready for offline (re)build.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

