from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]

PREPARE = REPO_ROOT / "scripts" / "tools" / "prepare_offline_bundle.py"
EXTRACT = REPO_ROOT / "scripts" / "tools" / "extract_offline_bundle.py"


def _load_script(path: Path, name: str) -> types.ModuleType:
    """Load a standalone script as a real module so internals can be patched.

    ``runpy.run_path`` returns a bare dict that is NOT the function globals, so
    patching ``_installed_bundle`` etc. via that dict silently has no effect.
    Loading through ``importlib`` yields a real module whose attributes are the
    true function globals and can be patched with ``mock.patch.object``.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare() -> types.ModuleType:
    return _load_script(PREPARE, "__test_offline_prepare")


def _load_extract() -> types.ModuleType:
    return _load_script(EXTRACT, "__test_offline_extract")


def _call_main(module: types.ModuleType, args: list[str]) -> int:
    """Call a script's main() with a controlled sys.argv."""
    with mock.patch.object(sys, "argv", ["script", *args]):
        return int(module.main())


def _make_cache(root: Path) -> Path:
    """Build a fake third-party cache mirroring the real repository layout."""
    cache = root / "third_party"
    # json: header-only, needs include/ for the CMake probe; single_include is non-build noise
    (cache / "json" / "include" / "nlohmann").mkdir(parents=True)
    (cache / "json" / "include" / "nlohmann" / "json.hpp").write_text("json", encoding="utf-8")
    (cache / "json" / "single_include" / "nlohmann").mkdir(parents=True)
    (cache / "json" / "single_include" / "nlohmann" / "json.hpp").write_text("json", encoding="utf-8")
    # eigen: header-only; Eigen/ is used by the build, bench/ docs are prunable noise
    (cache / "eigen" / "Eigen").mkdir(parents=True, exist_ok=True)
    (cache / "eigen" / "Eigen" / "Dense").write_text("eigen", encoding="utf-8")
    (cache / "eigen" / "bench").mkdir(parents=True, exist_ok=True)
    (cache / "eigen" / "bench" / "bench.cpp").write_text("bench", encoding="utf-8")
    # makeself: probed by makeself.sh / makeself-header.sh
    (cache / "makeself").mkdir(parents=True)
    (cache / "makeself" / "makeself.sh").write_text("makeself", encoding="utf-8")
    (cache / "makeself" / "makeself-header.sh").write_text("makeself", encoding="utf-8")
    # opbase: compiled from source; has a .git in the cache that must be stripped
    (cache / "opbase" / "src").mkdir(parents=True)
    (cache / "opbase" / "src" / "op.cc").write_text("opbase", encoding="utf-8")
    (cache / "opbase" / ".git").mkdir(parents=True)
    (cache / "opbase" / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    # catlass: probed by include/catlass/catlass.hpp
    (cache / "catlass" / "include" / "arch").mkdir(parents=True)
    (cache / "catlass" / "include" / "arch" / "arch.hpp").write_text("catlass", encoding="utf-8")
    (cache / "pkg").mkdir(parents=True)
    for name, data in {
        "abseil-cpp-20230802.1.tar.gz": b"abseil-arc",
        "protobuf-25.1.tar.gz": b"protobuf-arc",
        "include.zip": b"include-arc",
        "eigen-5.0.0.tar.gz": b"eigen-arc",
        "googletest-1.14.0.tar.gz": b"gtest-arc",
        "makeself-release-2.5.0-patch1.tar.gz": b"makeself-arc",
    }.items():
        (cache / "pkg" / name).write_bytes(data)
    return cache


class PrepareOfflineBundleTest(unittest.TestCase):
    """prepare_offline_bundle.py -- cache -> canonical offline bundle layout."""

    def test_produces_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(Path(tmp))
            out = Path(tmp) / "bundle"
            module = _load_prepare()
            rc = _call_main(module, ["--cache", str(cache), "--out", str(out)])
            self.assertEqual(rc, 0)
            # header-only / small components are shipped
            for rel in (
                "json/include/nlohmann/json.hpp",
                "eigen/Eigen/Dense",
                "makeself/makeself.sh",
                "makeself/makeself-header.sh",
                "opbase/src/op.cc",
                "catlass/include/arch/arch.hpp",
            ):
                self.assertTrue((out / rel).is_file(), f"missing {rel} in bundle")
            # abseil/protobuf ship only their pkg archives, NOT a pre-laid source dir
            self.assertFalse((out / "abseil-cpp").exists(), "abseil source dir should not be bundled")
            self.assertFalse((out / "protobuf").exists(), "protobuf source dir should not be bundled")
            self.assertFalse((out / "ascend_protobuf").exists())
            # pkg archives all present
            for rel in (
                "pkg/abseil-cpp-20230802.1.tar.gz",
                "pkg/protobuf-25.1.tar.gz",
                "pkg/include.zip",
                "pkg/eigen-5.0.0.tar.gz",
                "pkg/googletest-1.14.0.tar.gz",
                "pkg/makeself-release-2.5.0-patch1.tar.gz",
            ):
                self.assertTrue((out / rel).is_file(), f"missing {rel} in bundle")

    def test_prunes_non_build_noise_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(Path(tmp))
            out = Path(tmp) / "bundle"
            module = _load_prepare()
            self.assertEqual(_call_main(module, ["--cache", str(cache), "--out", str(out)]), 0)
            # json single_include is non-build noise and must be pruned
            self.assertFalse((out / "json" / "single_include").exists())
            # eigen bench/ is non-build noise and must be pruned
            self.assertFalse((out / "eigen" / "bench").exists())
            # build-critical content survives the prune
            self.assertTrue((out / "json" / "include" / "nlohmann" / "json.hpp").is_file())
            self.assertTrue((out / "eigen" / "Eigen" / "Dense").is_file())

    def test_strips_git_directories_from_copied_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = _make_cache(Path(tmp))
            out = Path(tmp) / "bundle"
            module = _load_prepare()
            self.assertEqual(_call_main(module, ["--cache", str(cache), "--out", str(out)]), 0)
            self.assertFalse((out / "opbase" / ".git").exists())
            self.assertFalse((out / "catlass" / ".git").exists())

    def test_missing_component_warns_but_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "third_party"
            (cache / "json").mkdir(parents=True)
            (cache / "json" / "x").write_text("x", encoding="utf-8")
            out = Path(tmp) / "bundle"
            module = _load_prepare()
            with mock.patch("sys.stdout") as stdout:
                rc = _call_main(module, ["--cache", str(cache), "--out", str(out)])
            self.assertEqual(rc, 0)
            warn_text = "".join(c[0][0] for c in stdout.write.call_args_list if c[0])
            self.assertIn("[warn]", warn_text)
            self.assertTrue((out / "json" / "x").is_file())

    def test_missing_cache_dir_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "does-not-exist"
            out = Path(tmp) / "bundle"
            module = _load_prepare()
            with self.assertRaises(SystemExit) as ctx:
                _call_main(module, ["--cache", str(cache), "--out", str(out)])
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertIn("cache directory not found", str(ctx.exception))


class ExtractOfflineBundleTest(unittest.TestCase):
    """extract_offline_bundle.py -- embedded bundle -> source-tree third_party/."""

    def _make_source_tree(self, root: Path) -> Path:
        src = root / "repo"
        src.mkdir(parents=True, exist_ok=True)
        (src / "CMakeLists.txt").write_text("cmake", encoding="utf-8")
        (src / "build.sh").write_text("build", encoding="utf-8")
        (src / "other.txt").write_text("keep", encoding="utf-8")
        (src / "third_party" / "abseil-cpp").mkdir(parents=True)
        (src / "third_party" / "abseil-cpp" / "OLD").write_text("old", encoding="utf-8")
        return src

    def _make_bundle(self, root: Path) -> Path:
        bundle = root / "bundle-root"
        (bundle / "abseil-cpp").mkdir(parents=True)
        (bundle / "abseil-cpp" / "NEW").write_text("new", encoding="utf-8")
        (bundle / "opbase" / "src").mkdir(parents=True)
        (bundle / "opbase" / "src" / "op.cc").write_text("opbase", encoding="utf-8")
        (bundle / "pkg").mkdir(parents=True)
        (bundle / "pkg" / "include.zip").write_bytes(b"arc")
        return bundle

    def test_missing_src_fails(self) -> None:
        module = _load_extract()
        with self.assertRaises(SystemExit) as ctx:
            _call_main(module, ["--src", "/nonexistent/repo"])
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("source tree not found", str(ctx.exception))

    def test_non_source_tree_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "not-a-repo"
            root.mkdir()
            (root / "CMakeLists.txt").write_text("x", encoding="utf-8")  # missing build.sh
            module = _load_extract()
            with self.assertRaises(SystemExit) as ctx:
                _call_main(module, ["--src", str(root)])
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertIn("does not look like a flash-linear-attention-npu source", str(ctx.exception))
            self.assertFalse((root / "third_party").exists())

    def test_bundle_absent_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._make_source_tree(root)
            module = _load_extract()
            with mock.patch.object(module, "_installed_bundle", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    _call_main(module, ["--src", str(src)])
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertIn(
                "could not locate the embedded offline bundle", str(ctx.exception)
            )
            # target third_party untouched when there is nothing embedded
            self.assertTrue((src / "third_party" / "abseil-cpp" / "OLD").is_file())

    def test_extracts_and_overwrites_third_party(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._make_source_tree(root)
            bundle = self._make_bundle(root)
            module = _load_extract()
            with mock.patch.object(module, "_installed_bundle", return_value=bundle):
                rc = _call_main(module, ["--src", str(src)])
            self.assertEqual(rc, 0)
            # overwritten by bundle contents
            self.assertFalse((src / "third_party" / "abseil-cpp" / "OLD").exists())
            self.assertTrue((src / "third_party" / "abseil-cpp" / "NEW").is_file())
            self.assertTrue((src / "third_party" / "opbase" / "src" / "op.cc").is_file())
            self.assertTrue((src / "third_party" / "pkg" / "include.zip").is_file())
            # unrelated files preserved
            self.assertTrue((src / "other.txt").is_file())


class OpbaseOfflineCMakeTest(unittest.TestCase):
    """cmake/third_party/opbase.cmake -- .git-aware handling of offline sources."""

    def test_git_conditional_branches_present(self) -> None:
        cmake = (REPO_ROOT / "cmake" / "third_party" / "opbase.cmake").read_text(encoding="utf-8")
        self.assertIn("${CANN_3RD_LIB_PATH}/opbase", cmake)
        self.assertIn('if(EXISTS "${OPBASE_SOURCE_PATH}/.git")', cmake)
        self.assertIn("COMMAND git checkout", cmake)
        self.assertIn("using bundled tarball sources as-is", cmake)


class OfflineBundleRoundTripTest(unittest.TestCase):
    """prepare -> extract keeps canonical layout stable end-to-end."""

    def test_roundtrip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = _make_cache(root)
            bundle = root / "bundle"
            prepare_mod = _load_prepare()
            self.assertEqual(
                _call_main(prepare_mod, ["--cache", str(cache), "--out", str(bundle)]), 0
            )

            src = root / "repo"
            src.mkdir(parents=True, exist_ok=True)
            (src / "CMakeLists.txt").write_text("cmake", encoding="utf-8")
            (src / "build.sh").write_text("build", encoding="utf-8")

            extract_mod = _load_extract()
            with mock.patch.object(extract_mod, "_installed_bundle", return_value=bundle):
                rc = _call_main(extract_mod, ["--src", str(src)])
            self.assertEqual(rc, 0)

            for rel in (
                "json/include/nlohmann/json.hpp",
                "eigen/Eigen/Dense",
                "makeself/makeself.sh",
                "catlass/include/arch/arch.hpp",
                "opbase/src/op.cc",
                "pkg/include.zip",
                "pkg/abseil-cpp-20230802.1.tar.gz",
                "pkg/protobuf-25.1.tar.gz",
            ):
                self.assertTrue((src / "third_party" / rel).is_file(), f"missing {rel} after round-trip")
            # abseil/protobuf remain archive-only (no pre-laid source dir) after round-trip
            self.assertFalse((src / "third_party" / "abseil-cpp").exists())
            self.assertFalse((src / "third_party" / "protobuf").exists())


if __name__ == "__main__":
    unittest.main()
