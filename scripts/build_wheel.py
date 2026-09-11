"""Build the root wheel and print its exact installation command."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))


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


def _runtime_pins() -> list[str]:
    """torch / torch_npu pins derived from the build environment."""

    pins: list[str] = []
    try:
        import torch

        pins.append(f"torch=={torch.__version__}")
    except Exception:
        pass
    try:
        import torch_npu

        pins.append(f"torch_npu=={torch_npu.__version__}")
    except Exception:
        pass
    return pins


def _prepare_abi_free_launcher() -> None:
    """Stage the ABI-free launcher and drop stale pybind leftovers.

    ``pip wheel`` builds in a temporary copy of the project, so preparing the
    package directory here (before the wheel is built) is what actually decides
    what ships: without it a stale ``_C_thin*.so`` from an earlier in-place build
    gets packaged as data and the wheel claims ``py3-none-any`` while holding a
    cpXXX binary.

    The default build is the ABI-free one: pure Python plus
    ``libfla_npu_thin.so``, no CPython ABI, no libtorch C++ ABI.  The pybind
    extension is opt-in (``FLA_NPU_BUILD_THIN=1``) for A/B comparisons, and a
    pure-ctypes wheel is ``FLA_NPU_BUILD_STABLE_ABI=0``.
    """

    package_dir = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu"
    if not package_dir.is_dir():
        return
    if os.getenv("FLA_NPU_BUILD_THIN", "FALSE").upper() not in {"1", "TRUE",
                                                              "YES", "ON"}:
        for pattern in ("_C_thin*.so", "_C_thin*.pyd"):
            for stale in package_dir.glob(pattern):
                stale.unlink()
                print(f"[fla-npu build] dropped stale {stale.name}", flush=True)
    if os.getenv("FLA_NPU_BUILD_STABLE_ABI", "TRUE").upper() in {
            "0", "FALSE", "NO", "OFF"}:
        return
    builder = (REPO_ROOT / "torch_custom" / "fla_npu" / "csrc_stable"
               / "build_stable.py")
    target = package_dir / "libfla_npu_thin.so"
    subprocess.run([sys.executable, str(builder), "--no-debug-probe",
                    "--out", str(target)], check=True)
    print(f"[fla-npu build] staged {target.name} ({target.stat().st_size} bytes)",
          flush=True)


def _inject_runtime_pins(wheel_path: Path) -> None:
    """Add Requires-Dist pins to a wheel that carries a compiled launcher.

    pyproject.toml owns ``[project]`` metadata, so ``install_requires`` in
    setup.py is ignored; the pins have to be injected into the produced wheel.
    They exist so pip refuses to install the ABI-matched extension next to a
    different torch/torch_npu instead of failing at import time.
    """

    with zipfile.ZipFile(wheel_path) as archive:
        infos = archive.infolist()
        blobs = {info.filename: archive.read(info.filename) for info in infos}

    if not any(name.endswith(".so") and "_C_thin" in name for name in blobs):
        return  # pure-python wheel: nothing to pin
    pins = _runtime_pins()
    if not pins:
        return
    meta_name = next(name for name in blobs
                     if name.endswith(".dist-info/METADATA"))
    meta = blobs[meta_name].decode("utf-8")
    if any(f"Requires-Dist: {pin}" in meta for pin in pins):
        return
    lines = meta.splitlines()
    insert_at = len(lines)
    for index, line in enumerate(lines):
        if line.startswith("Requires-Dist:"):
            insert_at = index + 1
    lines[insert_at:insert_at] = [f"Requires-Dist: {pin}" for pin in pins]
    blobs[meta_name] = ("\n".join(lines) + "\n").encode("utf-8")

    record_name = next(name for name in blobs if name.endswith(".dist-info/RECORD"))
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(blobs[meta_name]).digest()).rstrip(b"=").decode()
    size = len(blobs[meta_name])
    record = [
        f"{meta_name},sha256={digest},{size}"
        if line.startswith(meta_name + ",") else line
        for line in blobs[record_name].decode("utf-8").splitlines()
    ]
    blobs[record_name] = ("\n".join(record) + "\n").encode("utf-8")

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, blobs[info.filename])
    print(f"[fla-npu build] pinned {', '.join(pins)} into the wheel metadata",
          flush=True)


def _collect_build_args(args: argparse.Namespace) -> str:
    parts = list(args.build_args)
    env_args = os.getenv("FLA_NPU_BUILD_ARGS", "").strip()
    if env_args:
        parts.insert(0, env_args)
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_values(build_args: str, option: str) -> list:
    """Extract the comma-separated values of an option from the forwarded args."""
    values: list[str] = []
    tokens = build_args.split()
    for i, token in enumerate(tokens):
        if token == option:
            if i + 1 < len(tokens):
                values.extend(tokens[i + 1].split(","))
            continue
        if token.startswith(f"{option}="):
            values.extend(token.split("=", 1)[1].split(","))
    return values


def _drop_option(build_args: str, option: str) -> str:
    """Remove all occurrences of an option (space-separated or = spelling)."""
    tokens = build_args.split()
    filtered: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == option:
            i += 2
            continue
        if token.startswith(f"{option}="):
            i += 1
            continue
        filtered.append(token)
        i += 1
    return " ".join(filtered)


def _native_build_args(args: argparse.Namespace) -> list:
    """Map 一键编包的原生 -g / --sanitizer / --oom 选项到 asc_opc 合法值。

    - -g / --debug   -> ccec_g（kernel 调试信息）
    - --sanitizer    -> sanitizer（asc_opc 内存越界插桩，CANN 9.1.0 起合法）
    - --oom          -> oom（kernel 侧 OOM 检查）

    值通过 build.sh --bisheng_flags 或 --op_debug_config 传递，最终由
    ascendc_bin_param_build.py 拼成 asc_opc 的 --op_debug_config=<values>。
    CANN 9.1.0 的 asc_opc 合法值表为
    (oom, dump_cce, dump_bin, dump_loc, ccec_O0, ccec_g, check_flag, sanitizer)，
    因此 --sanitizer 必须映射为 sanitizer，不能使用更高版本才识别的
    check_flag_sanitizer。
    """
    configs = []
    if args.debug:
        configs.append("ccec_g")
    if args.sanitizer:
        configs.append("sanitizer")
    if args.oom:
        configs.append("oom")
    return configs


def _assemble_build_args(args: argparse.Namespace) -> str:
    build_args = _collect_build_args(args)
    native = _native_build_args(args)
    if not native:
        return build_args
    # 原生选项优先：将 --bisheng_flags 与 --op_debug_config 中已经存在的
    # 用户显式值全部取出，与原生值合并去重后只传一次，保留 dump_cce 等
    # 用户显式配置（review 之前的问题：直接丢弃用户的配置）。
    option = "--bisheng_flags"
    existing = (
        _extract_values(build_args, "--bisheng_flags")
        + _extract_values(build_args, "--op_debug_config")
    )
    build_args = _drop_option(build_args, "--bisheng_flags")
    build_args = _drop_option(build_args, "--op_debug_config")
    merged = list(dict.fromkeys(native + existing))
    # build.sh 只识别 --bisheng_flags=<values> 的等号写法，不能用空格分隔。
    tail = f"{option}={','.join(merged)}" if merged else ""
    return f"{build_args} {tail}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        default="dist",
        help="wheel output directory relative to the repository root (default: dist)",
    )
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help=(
            "add kernel debug info (asc_opc -g). Equivalent to passing "
            "--op_debug_config ccec_g to build.sh."
        ),
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help=(
            "enable Ascend kernel memory sanitizer support for mssanitizer. "
            "Maps to sanitizer (asc_opc --op_debug_config=sanitizer), which "
            "instruments kernels to detect memory errors. Runtime detection is "
            "done by mssanitizer via LD_PRELOAD injection "
            "(libmssanitizer_injection.so). Requires the Ascend toolkit's "
            "mssanitizer debug environment when running."
        ),
    )
    parser.add_argument(
        "--oom",
        action="store_true",
        help=(
            "enable kernel-side OOM debug. Maps to oom (asc_opc "
            "--op_debug_config=oom) for build.sh."
        ),
    )
    parser.add_argument(
        "--build-args",
        action="append",
        default=[],
        metavar="ARGS",
        help=(
            "extra arguments forwarded to build.sh (e.g. "
            "--build-args='-O3'). build.sh parses option values "
            "space-separated, so do not use '=' between an option and its "
            "value. May be repeated or space-separated within one value. "
            "Also honored via the FLA_NPU_BUILD_ARGS environment variable."
        ),
    )
    args = parser.parse_args()

    wheel_dir = _resolve_output_dir(args.wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _prepare_abi_free_launcher()
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

    env = os.environ.copy()
    build_args = _assemble_build_args(args)
    if build_args:
        env["FLA_NPU_BUILD_ARGS"] = build_args
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)

    # With the thin launcher enabled the wheel is no longer py3-none-any
    # (it embeds a cpXXX-linux_aarch64 extension), so resolve the actual file.
    wheel_files = sorted(wheel_dir.glob("flash_linear_attention_npu-*.whl"))
    if not wheel_files:
        raise RuntimeError(f"Expected wheel was not produced under {wheel_dir}")
    wheel_path = wheel_files[-1]

    _inject_runtime_pins(wheel_path)

    print(f"[fla-npu build] Wheel: {wheel_path}", flush=True)
    print(f"[fla-npu build] Install command:", flush=True)
    print(_install_command(wheel_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
