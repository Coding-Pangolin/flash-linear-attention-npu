"""Build the root wheel and print its exact installation command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fla_npu_artifacts import get_wheel_filename  # noqa: E402


def _resolve_output_dir(value: str) -> Path:
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    return output_dir.resolve()


def _install_command(wheel_path: Path) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m pip install "
        "--force-reinstall --no-cache-dir --no-deps "
        f"{shlex.quote(str(wheel_path))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        default="dist",
        help="wheel output directory relative to the repository root (default: dist)",
    )
    args = parser.parse_args()

    wheel_dir = _resolve_output_dir(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / get_wheel_filename(REPO_ROOT)

    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        ".",
        "-w",
        str(wheel_dir),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not wheel_path.is_file():
        raise RuntimeError(f"Expected wheel was not produced: {wheel_path}")

    print(f"[fla-npu build] Wheel: {wheel_path}", flush=True)
    print("[fla-npu build] Install command:", flush=True)
    print(_install_command(wheel_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
