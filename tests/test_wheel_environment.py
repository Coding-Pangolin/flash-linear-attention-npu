from __future__ import annotations

import csv
import importlib
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import setuptools
import zipfile


try:
    import packaging.version  # noqa: F401
except ImportError:
    from pip._vendor import packaging as vendored_packaging
    from pip._vendor.packaging import version as vendored_packaging_version

    sys.modules["packaging"] = vendored_packaging
    sys.modules["packaging.version"] = vendored_packaging_version


if not hasattr(importlib, "metadata"):
    # The project requires Python >=3.9. This compatibility shim only lets the
    # source-only packaging tests run on older maintenance hosts.
    metadata_module = types.ModuleType("importlib.metadata")

    def _missing_distribution(_name: str) -> str:
        raise RuntimeError("distribution metadata is unavailable")

    metadata_module.version = _missing_distribution
    importlib.metadata = metadata_module
    sys.modules["importlib.metadata"] = metadata_module


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    REPO_ROOT
    / "torch_custom"
    / "fla_npu"
    / "fla_npu"
    / "__init__.py"
)


def _load_setup() -> tuple[dict[str, object], dict[str, object]]:
    setup_kwargs: dict[str, object] = {}

    def capture_setup(**kwargs) -> None:
        setup_kwargs.update(kwargs)

    with mock.patch.object(setuptools, "setup", side_effect=capture_setup):
        setup_globals = runpy.run_path(str(REPO_ROOT / "setup.py"))
    return setup_globals, setup_kwargs


def _load_run_package_finalizer() -> dict[str, object]:
    return runpy.run_path(
        str(
            REPO_ROOT
            / "scripts"
            / "package"
            / "ops_transformer"
            / "scripts"
            / "finalize_wheel_opp.py"
        )
    )


def _create_minimal_vendor(vendor_dir: Path, *, include_alias: bool = False) -> None:
    required_files = (
        vendor_dir / "op_api" / "lib" / "libcust_opapi.so",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "op_host"
        / "lib"
        / "linux"
        / "x86_64"
        / "libophost.so",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "ascend910b"
        / "sample"
        / "sample.o",
        vendor_dir
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
        / "ascend910b"
        / "binary_info_config.json",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    if include_alias:
        (vendor_dir / "op_api" / "lib" / "libopapi.so").write_bytes(b"unsafe")


def _create_runtime_package(site_root: Path) -> tuple[Path, Path, Path]:
    package_dir = site_root / "fla_npu"
    package_dir.mkdir(parents=True)
    runtime_path = package_dir / "__init__.py"
    shutil.copy2(RUNTIME_SOURCE, runtime_path)
    vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
    packaged_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    packaged_opapi.parent.mkdir(parents=True)
    packaged_opapi.write_bytes(b"packaged-test")
    return runtime_path, vendor_dir, packaged_opapi


class WheelEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_globals, cls.setup_kwargs = _load_setup()

    def test_staging_removes_conflicting_libopapi_alias(self) -> None:
        stage_run_package = self.setup_globals["_stage_run_package"]

        with tempfile.TemporaryDirectory() as temp_dir:
            opp_root = Path(temp_dir) / "opp"

            def fake_install_run_package(_run_file, target_root) -> None:
                _create_minimal_vendor(
                    Path(target_root) / "vendors" / "fla_npu_transformer",
                    include_alias=True,
                )

            stage_run_package.__globals__["_install_run_package"] = fake_install_run_package
            stage_run_package(Path(temp_dir) / "unused.run", opp_root)

            lib_dir = (
                opp_root
                / "vendors"
                / "fla_npu_transformer"
                / "op_api"
                / "lib"
            )
            self.assertTrue((lib_dir / "libcust_opapi.so").is_file())
            self.assertFalse((lib_dir / "libopapi.so").exists())
            self.assertFalse((opp_root.parent / "_lib").exists())

            custom_build = (REPO_ROOT / "cmake" / "custom_build.cmake").read_text(
                encoding="utf-8"
            )
            symbol_build = (REPO_ROOT / "cmake" / "symbol.cmake").read_text(
                encoding="utf-8"
            )
            self.assertIn("NO_SONAME ON", custom_build)
            self.assertIn("NO_SONAME ON", symbol_build)

    def test_package_build_always_starts_from_clean_outputs(self) -> None:
        setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        build_source = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")

        self.assertNotIn("FLA_NPU_INCREMENTAL_BUILD", setup_source)
        self.assertNotIn("FLA_NPU_SKIP_RUN_BUILD", setup_source)
        self.assertNotIn("FLA_NPU_SKIP_RUN_INSTALL", setup_source)
        self.assertNotIn("--incremental", build_source)
        self.assertIn("set_env\n\nclean\nclean_build_out", build_source)

    def test_package_build_supports_single_op_filter(self) -> None:
        setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("FLA_NPU_OPS", setup_source)
        self.assertIn("--ops=", setup_source)

    def test_default_build_is_the_abi_free_one(self) -> None:
        """The default artifact must not pin the CPython/libtorch C++ ABI.

        The stable launcher is plain package data and the legacy extension is
        off, so no CPython ABI is involved -- but the payload is still built for
        one host platform, which is what the wheel has to say about itself.
        """

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLA_NPU_BUILD_STABLE_ABI", None)
            setup_globals, setup_kwargs = _load_setup()
            # The predicate reads os.environ, so it has to be called while the
            # environment under test is still in place.
            self.assertTrue(setup_globals["_stable_build_enabled"]())
            distribution = setup_kwargs["distclass"]()
            self.assertFalse(distribution.is_pure())
            self.assertFalse(distribution.has_ext_modules())

    def test_default_wheel_is_tagged_by_platform_not_by_python(self) -> None:
        """py3-none-<platform>: one wheel per host, not one per Python minor.

        ``any`` would let pip install an aarch64 launcher on x86_64, and a
        cp3xx tag would multiply the release matrix for a wheel that carries no
        CPython extension at all.
        """

        import sysconfig

        setup_globals, setup_kwargs = _load_setup()
        command = setup_globals["CMDCLASS"]["bdist_wheel"](
            setup_kwargs["distclass"]({"name": "flash-linear-attention-npu",
                                       "version": "0"}))
        command.finalize_options()
        python, abi, platform = command.get_tag()
        self.assertEqual((python, abi), ("py3", "none"))
        self.assertNotEqual(platform, "any")
        self.assertEqual(
            platform,
            sysconfig.get_platform().replace("-", "_").replace(".", "_"))

    def test_stable_launcher_can_be_turned_off(self) -> None:
        with mock.patch.dict(os.environ,
                             {"FLA_NPU_BUILD_STABLE_ABI": "0"}):
            setup_globals, _ = _load_setup()
            self.assertFalse(setup_globals["_stable_build_enabled"]())

    @staticmethod
    def _artifacts_globals() -> dict[str, object]:
        return runpy.run_path(str(REPO_ROOT / "scripts" / "fla_npu_artifacts.py"))

    @staticmethod
    def _public_version() -> str:
        text = (REPO_ROOT / "fla" / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.MULTILINE)
        assert match is not None, "fla/__init__.py carries no __version__"
        return match.group(1)

    def test_pypi_wheel_is_named_after_the_tier_and_the_arch(self) -> None:
        """One project per tier, one platform tag per arch, no build tag.

        pip picks the artifact from the distribution name and the wheel tags, so
        a published name has to say both: the tier (which SoC the payload serves)
        and the arch (which host libraries it carries).
        """

        artifacts = self._artifacts_globals()
        version = self._public_version()
        for soc, tier in (("ascend910b", "a2"),
                          ("ascend910_93", "a3"),
                          ("ascend950", "a5")):
            for arch in ("aarch64", "x86_64"):
                with mock.patch.dict(os.environ, {
                        "FLA_NPU_PYPI": "TRUE",
                        "FLA_NPU_SOC": soc,
                        "FLA_NPU_ARCH": arch}):
                    self.assertEqual(artifacts["get_tier"](), tier)
                    self.assertEqual(
                        artifacts["get_distribution_name"](),
                        f"flash-linear-attention-npu-{tier}")
                    self.assertEqual(
                        artifacts["get_wheel_filename"](REPO_ROOT),
                        f"flash_linear_attention_npu_{tier}-{version}-py3-none-"
                        f"manylinux_2_28_{arch}.whl")

    def test_pypi_wheel_keeps_the_base_name_outside_pypi_mode(self) -> None:
        artifacts = self._artifacts_globals()
        with mock.patch.dict(os.environ, {
                "FLA_NPU_SOC": "ascend950",
                "FLA_NPU_ARCH": "x86_64"}):
            os.environ.pop("FLA_NPU_PYPI", None)
            self.assertEqual(artifacts["get_distribution_name"](),
                             "flash-linear-attention-npu")
            if sys.platform.startswith("linux"):
                self.assertTrue(
                    artifacts["get_wheel_filename"](REPO_ROOT).startswith(
                        "flash_linear_attention_npu-"))

    def test_pypi_file_name_and_wheel_tag_agree(self) -> None:
        """The upload name and the METADATA tag come from two places.

        ``get_wheel_filename`` predicts the artifact for CI, while bdist_wheel
        writes the tag from ``get_tag``.  If they disagree, pip installs an
        aarch64 payload on x86_64 (or PyPI rejects linux_<arch> outright), which
        is only discovered after upload.
        """

        artifacts = self._artifacts_globals()
        for arch in ("aarch64", "x86_64"):
            with mock.patch.dict(os.environ, {
                    "FLA_NPU_PYPI": "TRUE",
                    "FLA_NPU_SOC": "ascend950",
                    "FLA_NPU_ARCH": arch}):
                setup_globals, setup_kwargs = _load_setup()
                command = setup_globals["CMDCLASS"]["bdist_wheel"](
                    setup_kwargs["distclass"]({
                        "name": "flash-linear-attention-npu-a5",
                        "version": "0"}))
                command.finalize_options()
                python, abi, platform = command.get_tag()
                self.assertEqual((python, abi), ("py3", "none"))
                # PyPI accepts PEP 600 tags only, so this must not be
                # linux_<arch> (the sysconfig spelling).
                self.assertEqual(platform, f"manylinux_2_28_{arch}")
                self.assertTrue(
                    artifacts["get_wheel_filename"](REPO_ROOT).endswith(
                        f"-{platform}.whl"))

    def test_tier_metadata_is_generated_for_pypi_wheels_only(self) -> None:
        """The wheel records its tier, and a local build carries no marker.

        scripts/check_pypi_wheel.py reads TIER to catch a wheel published under
        the wrong project, and the version promise is generated from
        scripts/npu_compat.py so the guard cannot drift from the README.
        """

        package_dir = REPO_ROOT / "torch_custom" / "fla_npu" / "fla_npu"
        build_meta = package_dir / "_build_meta.py"
        compat_py = package_dir / "_compat.py"
        try:
            with mock.patch.dict(os.environ, {
                    "FLA_NPU_PYPI": "TRUE",
                    "FLA_NPU_SOC": "ascend950"}):
                setup_globals, _ = _load_setup()
                setup_globals["_write_runtime_meta"]()
            self.assertIn("TIER = 'a5'", build_meta.read_text(encoding="utf-8"))
            compat = compat_py.read_text(encoding="utf-8")
            self.assertIn("MIN_CANN = '9.0.0'", compat)
            self.assertIn("MIN_TORCH = '2.7.1'", compat)

            with mock.patch.dict(os.environ, {}):
                os.environ.pop("FLA_NPU_PYPI", None)
                setup_globals["_write_runtime_meta"]()
            self.assertFalse(build_meta.exists())
            self.assertFalse(compat_py.exists())
        finally:
            build_meta.unlink(missing_ok=True)
            compat_py.unlink(missing_ok=True)

    def _write_release_wheel(self, path: Path, *, tier: str, soc: str,
                             declared_tier: str | None = None) -> None:
        """A minimal wheel with the file name, metadata and payload the gate reads."""

        info = f"flash_linear_attention_npu_{tier}-26.9.0.dist-info"
        vendor = "fla_npu/opp/vendors/fla_npu_transformer"
        entries = {
            "fla_npu/libfla_npu_stable.so": b"\x7fELF stable",
            "fla_npu/ops/ascendc/_stable_hash.py": b"SOURCE_HASH = 'deadbeef'\n",
            "fla_npu/_build_meta.py": f"TIER = {declared_tier or tier!r}\n".encode(),
            "fla_npu/_compat.py": b"MIN_CANN = '8.5.2'\nMIN_TORCH = '2.7.1'\n",
            f"{vendor}/op_api/lib/libcust_opapi.so": b"\x7fELF opapi",
            f"{vendor}/op_impl/ai_core/tbe/kernel/{soc}/demo/demo.o": b"kernel",
            f"{vendor}/op_impl/ai_core/tbe/config/{soc}/aic-ops-info.json": b"{}",
        }
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{info}/METADATA", (
                "Metadata-Version: 2.1\n"
                f"Name: flash-linear-attention-npu-{tier}\n"
                "Version: 26.9.0\n"
                "Requires-Python: >=3.9\n"
                "Requires-Dist: torch>=2.7.1\n"
                "Requires-Dist: torch_npu>=2.7.1\n"))
            for name, payload in entries.items():
                archive.writestr(name, payload)
            archive.writestr(f"{info}/RECORD",
                             f"{info}/METADATA,,\n{info}/RECORD,,\n")

    def test_release_gate_rejects_a_mislabelled_tiered_wheel(self) -> None:
        """The upload gate has to fail on the wheels a user would trip over.

        Its ELF checks need readelf, so the structural checks are exercised here
        and the ELF ones run against the real wheel in CI (and locally on Linux).
        """

        gate = runpy.run_path(str(REPO_ROOT / "scripts" / "check_pypi_wheel.py"))

        def run_gate(wheel: Path):
            return gate["check_wheel"](
                wheel,
                expect_tier="a2",
                expect_arch="aarch64",
                expect_version="26.9.0",
                require_offline_bundle=False,
                require_launcher=True,
                max_glibc="2.28",
                max_glibcxx="3.4.29",
                allow_missing_readelf=True,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir) / (
                "flash_linear_attention_npu_a2-26.9.0-py3-none-"
                "manylinux_2_28_aarch64.whl")
            self._write_release_wheel(wheel, tier="a2", soc="ascend910b")
            with mock.patch("shutil.which", return_value=None):
                notes = run_gate(wheel)
            self.assertTrue(any("kernel" in note for note in notes), notes)

            # A wheel whose tier marker contradicts its name installs under the
            # wrong project and serves the wrong SoC.
            self._write_release_wheel(wheel, tier="a2", soc="ascend910b",
                                      declared_tier="a5")
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaisesRegex(gate["CheckFailure"], "TIER"):
                    run_gate(wheel)

            # Kernels for another SoC: the file name promises a2, the payload
            # carries 950 kernels.
            self._write_release_wheel(wheel, tier="a2", soc="ascend950")
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaisesRegex(gate["CheckFailure"], "kernels"):
                    run_gate(wheel)

    def test_pure_ctypes_wheel_is_not_required_to_carry_the_launcher(self) -> None:
        gate = runpy.run_path(str(REPO_ROOT / "scripts" / "check_pypi_wheel.py"))
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir) / (
                "flash_linear_attention_npu_a5-26.9.0-py3-none-"
                "manylinux_2_28_x86_64.whl")
            self._write_release_wheel(wheel, tier="a5", soc="ascend950")
            with zipfile.ZipFile(wheel) as archive:
                payload = {i.filename: archive.read(i.filename)
                           for i in archive.infolist()}
            payload.pop("fla_npu/libfla_npu_stable.so")
            payload.pop("fla_npu/ops/ascendc/_stable_hash.py")
            with zipfile.ZipFile(wheel, "w") as archive:
                for name, blob in payload.items():
                    archive.writestr(name, blob)
            with mock.patch("shutil.which", return_value=None):
                gate["check_wheel"](
                    wheel,
                    expect_tier="a5",
                    expect_arch="x86_64",
                    expect_version="26.9.0",
                    require_offline_bundle=False,
                    require_launcher=False,
                    max_glibc="2.28",
                    max_glibcxx="3.4.29",
                    allow_missing_readelf=True,
                )

    def _write_minimal_wheel(self, path: Path, entries: dict) -> None:
        info = "demo-1.0.dist-info"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{info}/METADATA", (
                "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
                "Requires-Dist: numpy\n"))
            for name, payload in entries.items():
                archive.writestr(name, payload)
            archive.writestr(f"{info}/RECORD",
                             f"{info}/METADATA,,\n{info}/RECORD,,\n")

    @staticmethod
    def _wheel_metadata(wheel: Path) -> str:
        with zipfile.ZipFile(wheel) as archive:
            name = next(entry for entry in archive.namelist()
                        if entry.endswith("METADATA"))
            return archive.read(name).decode("utf-8")

    def test_abi_free_wheel_declares_a_torch_lower_bound(self) -> None:
        """One wheel for every torch above the floor, not an exact pin.

        The Stable-ABI launcher resolves its aoti_torch_* symbols at load, so
        what it needs is a minimum version rather than the exact build it was
        compiled against.
        """

        build_wheel = runpy.run_path(str(REPO_ROOT / "scripts" / "build_wheel.py"))
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir) / "demo-1.0-py3-none-any.whl"
            self._write_minimal_wheel(
                wheel, {"fla_npu/libfla_npu_stable.so": b"\x7fELF"})
            build_wheel["_inject_runtime_pins"](wheel)
            text = self._wheel_metadata(wheel)
        self.assertIn("Requires-Dist: torch>=2.7.1", text)
        self.assertIn("Requires-Dist: torch_npu>=2.7.1", text)
        self.assertNotIn("Requires-Dist: torch==", text)

    def test_pure_ctypes_wheel_declares_nothing(self) -> None:
        """No compiled launcher means no torch constraint to add."""

        build_wheel = runpy.run_path(str(REPO_ROOT / "scripts" / "build_wheel.py"))
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir) / "demo-1.0-py3-none-any.whl"
            self._write_minimal_wheel(wheel, {"fla_npu/__init__.py": b""})
            before = self._wheel_metadata(wheel)
            build_wheel["_inject_runtime_pins"](wheel)
            after = self._wheel_metadata(wheel)
        self.assertEqual(before, after)

    def test_generated_set_env_is_idempotent(self) -> None:
        rewrite_set_env = self.setup_globals["_rewrite_set_env"]

        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = (
                Path(temp_dir) / "opp" / "vendors" / "fla_npu_transformer"
            )
            rewrite_set_env(vendor_dir)
            set_env = vendor_dir / "bin" / "set_env.bash"
            script = f"""
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source {set_env!s}
source {set_env!s}
[[ "${{ASCEND_CUSTOM_OPP_PATH}}" == "{vendor_dir.parent.parent}:{vendor_dir}" ]]
[[ -z "${{LD_LIBRARY_PATH-}}" ]]
[[ "${{FLA_NPU_OPP_PATH}}" == "{vendor_dir.parent.parent}" ]]
[[ "${{FLA_NPU_OP_API_LIB}}" == "{vendor_dir}/op_api/lib/libcust_opapi.so" ]]
"""
            subprocess.run(["bash", "-c", script], check=True)

    def test_external_opp_install_removes_conflicting_alias(self) -> None:
        install_opp_globals = runpy.run_path(
            str(
                REPO_ROOT
                / "torch_custom"
                / "fla_npu"
                / "fla_npu"
                / "install_opp.py"
            )
        )
        install_opp = install_opp_globals["install_opp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_opp_root = temp_root / "package-opp"
            vendor_src = (
                package_opp_root
                / "vendors"
                / "fla_npu_transformer"
            )
            _create_minimal_vendor(vendor_src, include_alias=True)
            install_opp.__globals__["PACKAGE_OPP_ROOT"] = package_opp_root

            install_root = temp_root / "external-opp"
            install_opp(install_root)
            installed_vendor = (
                install_root
                / "vendors"
                / "fla_npu_transformer"
            )
            self.assertTrue(
                (installed_vendor / "op_api" / "lib" / "libcust_opapi.so").is_file()
            )
            self.assertFalse(
                (installed_vendor / "op_api" / "lib" / "libopapi.so").exists()
            )

    def test_run_package_overlay_finalization_is_idempotent(self) -> None:
        finalize_wheel_opp = _load_run_package_finalizer()["finalize_wheel_opp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir) / "site-packages"
            package_dir = site_root / "fla_npu"
            vendor_dir = (
                package_dir / "opp" / "vendors" / "fla_npu_transformer"
            )
            _create_minimal_vendor(vendor_dir, include_alias=True)
            config = package_dir / "opp" / "vendors" / "config.ini"
            config.write_text("load_priority=fla_npu_transformer\n", encoding="utf-8")

            dist_info = site_root / "flash_linear_attention_npu-1.0.dist-info"
            dist_info.mkdir(parents=True)
            record = dist_info / "RECORD"
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(
                    [
                        ["fla_npu/__init__.py", "", ""],
                        ["fla_npu/opp/stale-from-older-overlay.json", "", ""],
                        [f"{dist_info.name}/RECORD", "", ""],
                    ]
                )

            finalize_wheel_opp(package_dir)
            first_record = record.read_bytes()
            first_set_env = (vendor_dir / "bin" / "set_env.bash").read_bytes()
            finalize_wheel_opp(package_dir)

            self.assertEqual(record.read_bytes(), first_record)
            self.assertEqual(
                (vendor_dir / "bin" / "set_env.bash").read_bytes(),
                first_set_env,
            )
            self.assertTrue(
                (vendor_dir / "op_api" / "lib" / "libcust_opapi.so").is_file()
            )
            self.assertFalse((package_dir / "_lib").exists())
            self.assertFalse(
                (vendor_dir / "op_api" / "lib" / "libopapi.so").exists()
            )

            with record.open("r", encoding="utf-8", newline="") as handle:
                recorded_paths = {row[0] for row in csv.reader(handle) if row}
            current_opp_paths = {
                path.relative_to(site_root).as_posix()
                for path in (package_dir / "opp").rglob("*")
                if path.is_file() or path.is_symlink()
            }
            self.assertTrue(current_opp_paths.issubset(recorded_paths))
            self.assertNotIn(
                "fla_npu/opp/stale-from-older-overlay.json",
                recorded_paths,
            )
            self.assertIn("fla_npu/__init__.py", recorded_paths)

            set_env = vendor_dir / "bin" / "set_env.bash"
            script = f"""
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source {set_env!s}
source {set_env!s}
[[ "${{ASCEND_CUSTOM_OPP_PATH}}" == "{vendor_dir.parent.parent}:{vendor_dir}" ]]
[[ -z "${{LD_LIBRARY_PATH-}}" ]]
[[ "${{FLA_NPU_OPP_PATH}}" == "{vendor_dir.parent.parent}" ]]
[[ "${{FLA_NPU_OP_API_LIB}}" == "{vendor_dir}/op_api/lib/libcust_opapi.so" ]]
"""
            subprocess.run(["bash", "-c", script], check=True)

            dynamic_dir = (
                vendor_dir
                / "op_impl"
                / "ai_core"
                / "tbe"
                / "fla_npu_transformer_impl"
                / "dynamic"
            )
            dynamic_dir.mkdir(parents=True)
            source = dynamic_dir / "sample.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            bytecode = dynamic_dir / "__pycache__" / "sample.cpython-311.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"generated-by-pip")
            finalize_wheel_opp(package_dir)

            bytecode_relative = bytecode.relative_to(site_root).as_posix()
            with record.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            for row in rows:
                if row and row[0] == bytecode_relative:
                    row[1:] = ["", ""]
            with record.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(rows)

            checker = runpy.run_path(
                str(REPO_ROOT / "scripts" / "check_install_workflows.py")
            )
            checker["_assert_record_covers_opp"](package_dir)
            self.assertNotIn(
                bytecode_relative,
                checker["_manifest_from_directory"](package_dir),
            )

    def test_both_run_package_installers_finalize_wheel_overlay(self) -> None:
        installers = (
            REPO_ROOT / "cmake" / "scripts" / "custom" / "install.sh",
            REPO_ROOT
            / "scripts"
            / "package"
            / "ops_transformer"
            / "scripts"
            / "install.sh",
        )
        for installer in installers:
            with self.subTest(installer=installer):
                source = installer.read_text(encoding="utf-8")
                self.assertIn('finalize_wheel_opp "${wheel_opp_root}"', source)
                self.assertNotIn(
                    '"${dst_vendor}/op_api/lib/libopapi.so"',
                    source,
                )

        custom_build = (REPO_ROOT / "cmake" / "custom_build.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("finalize_wheel_opp.py", custom_build)

    def test_legacy_extension_rpath_keeps_standard_vendor_layout(self) -> None:
        setup_source = (
            REPO_ROOT / "torch_custom" / "fla_npu" / "setup.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("-Wl,-rpath,$ORIGIN/_lib", setup_source)
        self.assertIn(
            "-Wl,-rpath,$ORIGIN/opp/vendors/fla_npu_transformer/op_api/lib",
            setup_source,
        )

    def test_public_environment_diagnostic_recognizes_packaged_opapi(self) -> None:
        collector = runpy.run_path(str(REPO_ROOT / "scripts" / "collect_public_env.py"))
        has_op_api = collector["_has_op_api"]

        with tempfile.TemporaryDirectory() as temp_dir:
            packaged_opapi = (
                Path(temp_dir)
                / "fla_npu"
                / "opp"
                / "vendors"
                / "fla_npu_transformer"
                / "op_api"
                / "lib"
                / "libcust_opapi.so"
            )
            packaged_opapi.parent.mkdir(parents=True)
            packaged_opapi.write_bytes(b"packaged-test")
            with mock.patch.dict(
                os.environ,
                {"FLA_NPU_OP_API_LIB": str(packaged_opapi)},
                clear=True,
            ):
                self.assertTrue(has_op_api(Path(temp_dir) / "empty-opp"))

    def test_import_requires_cann_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CANN environment is not initialized"):
                runpy.run_path(str(RUNTIME_SOURCE))

    def test_import_loads_runtime_and_removes_stale_libopapi_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, vendor_dir, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            stale_alias = vendor_dir / "op_api" / "lib" / "libopapi.so"
            stale_alias.parent.mkdir(parents=True, exist_ok=True)
            stale_alias.write_bytes(b"unsafe")
            external_vendor = Path(temp_dir) / "cann" / "vendors" / "other"
            _create_minimal_vendor(external_vendor)

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                    "FLA_NPU_OPP_PATH": str(external_vendor),
                    "ASCEND_CUSTOM_OPP_PATH": str(external_vendor),
                    "ASCEND_OPP_PATH": str(external_vendor.parent.parent),
                },
                clear=True,
            ):
                def fake_cdll(path, *, mode):
                    del mode
                    if str(path) == "libopapi.so":
                        return mock.sentinel.cann_opapi
                    return mock.sentinel.custom_opapi

                with mock.patch("ctypes.CDLL", side_effect=fake_cdll) as cdll:
                    with self.assertWarnsRegex(RuntimeWarning, "Removed a stale"):
                        runtime_globals = runpy.run_path(str(runtime_path))
                    first = runtime_globals["load_ascendc_opapi_libraries"]()
                    second = runtime_globals["load_ascendc_opapi_libraries"]()
                    self.assertIs(first, second)
                    self.assertEqual(
                        first,
                        [mock.sentinel.custom_opapi, mock.sentinel.cann_opapi],
                    )

                    self.assertEqual(len(cdll.call_args_list), 2)
                    cann_call, custom_call = cdll.call_args_list
                    cann_args, cann_kwargs = cann_call
                    custom_args, custom_kwargs = custom_call
                    self.assertEqual(cann_args, ("libopapi.so",))
                    self.assertEqual(
                        custom_args,
                        (str(packaged_opapi),),
                    )
                    expected_mode = (
                        getattr(os, "RTLD_LOCAL", 0)
                        | getattr(os, "RTLD_NOW", 0)
                        | getattr(os, "RTLD_NODELETE", 0)
                    )
                    self.assertEqual(cann_kwargs["mode"], expected_mode)
                    self.assertEqual(custom_kwargs["mode"], expected_mode)
                    self.assertEqual(
                        expected_mode & getattr(os, "RTLD_GLOBAL", 0),
                        0,
                    )
                    self.assertEqual(
                        os.environ["ASCEND_CUSTOM_OPP_PATH"],
                        os.pathsep.join(
                            [
                                str(vendor_dir.parent.parent),
                                str(vendor_dir),
                                str(external_vendor),
                            ]
                        ),
                    )
                    self.assertEqual(
                        os.environ["ASCEND_OPP_PATH"],
                        str(external_vendor.parent.parent),
                    )
                    self.assertNotIn("LD_LIBRARY_PATH", os.environ)
                    self.assertEqual(
                        os.environ["FLA_NPU_OP_API_LIB"],
                        str(packaged_opapi),
                    )
            self.assertFalse(
                (vendor_dir / "op_api" / "lib" / "libopapi.so").exists()
            )

    def test_import_reports_missing_cann_opapi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, _ = _create_runtime_package(Path(temp_dir) / "site-packages")

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                },
                clear=True,
            ):
                with mock.patch("ctypes.CDLL", side_effect=OSError("not found")):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"Unable to load the CANN op_api library.*source.*CANN set_env\.sh",
                    ):
                        runpy.run_path(str(runtime_path))

    def test_import_accepts_packaged_opapi_in_standard_vendor_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )

            with mock.patch.dict(
                os.environ,
                {"ASCEND_HOME_PATH": "/fake/cann"},
                clear=True,
            ):
                with mock.patch(
                    "ctypes.CDLL",
                    side_effect=(mock.sentinel.cann_opapi, mock.sentinel.custom_opapi),
                ) as cdll:
                    runtime_globals = runpy.run_path(str(runtime_path))
                    self.assertEqual(
                        runtime_globals["load_ascendc_opapi_libraries"](),
                        [mock.sentinel.custom_opapi, mock.sentinel.cann_opapi],
                    )
                    self.assertEqual(
                        Path(cdll.call_args_list[1][0][0]).resolve(),
                        packaged_opapi.resolve(),
                    )

    def test_import_does_not_fallback_to_external_custom_opp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path, _, packaged_opapi = _create_runtime_package(
                Path(temp_dir) / "site-packages"
            )
            packaged_opapi.unlink()
            external_vendor = Path(temp_dir) / "cann" / "vendors" / "external"
            _create_minimal_vendor(external_vendor)

            with mock.patch.dict(
                os.environ,
                {
                    "ASCEND_HOME_PATH": "/fake/cann",
                    "FLA_NPU_OPP_PATH": str(external_vendor),
                    "ASCEND_CUSTOM_OPP_PATH": str(external_vendor),
                    "ASCEND_OPP_PATH": str(external_vendor.parent.parent),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"packaged FLA NPU op_api library",
                ):
                    runpy.run_path(str(runtime_path))

if __name__ == "__main__":
    unittest.main()
