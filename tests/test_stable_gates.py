"""Tests for the Stable-ABI gates themselves.

A gate that quietly stops working is worse than no gate: it still prints OK
while nothing is checked.  These tests are offline (no torch, no NPU) and cover
both directions -- the gates pass on the tree as it stands, and they fail on an
input that is deliberately wrong.

Usage:  python -m unittest tests.test_stable_gates
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = REPO_ROOT / "torch_custom" / "fla_npu"
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
GENERATED_INC = SETUP_DIR / "csrc_stable" / "generated" / "ops_stable_generated.inc"
GENERATED_PY = OPS_DIR / "_stable_generated.py"
SPEC_DIR = SETUP_DIR / "op_specs"


def _load_tool(name: str) -> dict:
    path = SETUP_DIR / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratedArtifactTest(unittest.TestCase):
    """The .inc and the Python glue come from one codegen run or from none."""

    def test_glue_hash_matches_the_checked_in_adapters(self) -> None:
        glue = GENERATED_PY.read_text(encoding="utf-8")
        match = re.search(r'^_GENERATED_HASH = "([0-9a-f]{32})"$', glue,
                          re.MULTILINE)
        self.assertIsNotNone(match, "glue carries no _GENERATED_HASH")
        # Newline-normalised: the file is CRLF in a Windows checkout and LF
        # elsewhere, and the stamp must not depend on that.
        expected = hashlib.md5(
            GENERATED_INC.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        self.assertEqual(
            match.group(1), expected,
            "ops_stable_generated.inc and _stable_generated.py are out of sync; "
            "re-run python tools/op_stable_codegen.py --all")

    def test_every_spec_has_a_stable_adapter(self) -> None:
        inc = GENERATED_INC.read_text(encoding="utf-8")
        generated = set(re.findall(r"kSchema_(npu_[a-z0-9_]+)\s*=", inc))
        hand_written = {
            name for name in re.findall(r"^def\s+(npu_[a-z0-9_]+)\s*\(",
                                        (OPS_DIR / "_stable.py").read_text(
                                            encoding="utf-8"),
                                        re.MULTILINE)
            if re.search(rf'^def {name}\(', inc, re.MULTILINE) is None
        }
        specs = {}
        for path in sorted(SPEC_DIR.glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            specs[spec["python_name"]] = path.name
        missing = sorted(set(specs) - generated - hand_written)
        self.assertEqual(missing, [], f"specs without an adapter: {missing}")
        orphan = sorted(generated - set(specs))
        self.assertEqual(orphan, [], f"adapters without a spec: {orphan}")

    def test_codegen_rejects_mismatched_stack_indices(self) -> None:
        """The check that would have caught the A5 segfault, exercised."""

        codegen = _load_tool("op_stable_codegen.py")
        lines = [f"  const bool flag{i} = to<bool>(stack[{i}]);"
                 for i in range(15)]
        # stream read from the slot before the end: compiles, then launches on a
        # garbage stream.
        bad = lines + ["  const int64_t stream = to<int64_t>(stack[14]);"]
        bad.append("  stack[0] = from(outputs[0]);")
        with self.assertRaises(ValueError) as caught:
            codegen._check_stack_indices("demo", bad, 15, 1)
        self.assertIn("0..15", str(caught.exception))

        good = lines + ["  const int64_t stream = to<int64_t>(stack[15]);"]
        good.append("  stack[0] = from(outputs[0]);")
        codegen._check_stack_indices("demo", good, 15, 1)


class GateCommandTest(unittest.TestCase):
    """The command-line gates pass on the current tree."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(SETUP_DIR), capture_output=True, text=True, check=False)

    def test_coverage_gate_is_green(self) -> None:
        result = self._run("tools/stable_coverage.py", "--strict")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALL COVERED", result.stdout)

    def test_api_parity_gate_is_green(self) -> None:
        result = self._run("tools/op_api_parity.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SIGNATURES MATCH", result.stdout)

    def test_abi_parity_gate_is_green(self) -> None:
        result = self._run("tools/op_abi_parity.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ABI MATCH", result.stdout)

    def test_abi_parity_gate_flags_a_changed_aclnn_signature(self) -> None:
        """The check that would have caught upstream #390, exercised.

        #390 turned four aclIntArray metadata slots into aclTensors and swapped
        an activationMode int for a const char*; nothing in the tree noticed
        until a kernel call died.  Here the implementation is edited the same
        way and the gate has to report it.
        """

        module = _load_tool("op_abi_parity.py")
        source = (OPS_DIR.parent.parent.parent / "fla_npu" / "ops" / "ascendc"
                  / "_aclnn_ctypes.py").read_text(encoding="utf-8")
        self.assertIn("ctx.int_array(cu_seqlens)", source)
        edited = source.replace("ctx.int_array(cu_seqlens)",
                                "ctx.tensor(cu_seqlens, 'cu')", 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "_aclnn_ctypes.py"
            reference.write_text(edited, encoding="utf-8")
            saved = module.REFERENCE
            module.REFERENCE = reference
            try:
                report = module.evaluate()
            finally:
                module.REFERENCE = saved
        flagged = {row["op"] for row in report["rows"] if row["problems"]}
        self.assertTrue(flagged, "a changed aclnn signature was not reported")
        detail = " ".join(problem for row in report["rows"]
                          for problem in row["problems"])
        self.assertIn("aclnn arguments differ", detail)

    def test_spec_python_blocks_are_synced(self) -> None:
        result = self._run("tools/sync_spec_python.py", "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("specs with drift: 0", result.stdout)

    def test_coverage_gate_fails_when_a_scenario_axis_is_impossible(self) -> None:
        """A declaration the spec cannot represent must not pass."""

        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = Path(temp_dir) / "aclnn_solve_tri.json"
            source = json.loads((SPEC_DIR / "aclnn_solve_tri.json").read_text(
                encoding="utf-8"))
            source["scenarios"] = {"layout": ["not-a-layout"]}
            spec_path.write_text(json.dumps(source), encoding="utf-8")
            module = _load_tool("stable_coverage.py")
            original = module.SPEC_DIR
            module.SPEC_DIR = Path(temp_dir)
            try:
                report = module.evaluate()
            finally:
                module.SPEC_DIR = original
        problems = [p for row in report["rows"] for p in row["problems"]]
        self.assertTrue(
            any("not-a-layout" in p for p in problems),
            f"an impossible scenario value was accepted: {problems}")

    def test_coverage_gate_records_integer_axes(self) -> None:
        """Integer knobs are declared rather than derived.

        conv1d's legal domain is not visible in its argument *kinds*:
        ``run_mode`` switches prefill/update, ``head_num`` switches the output
        reshape, and ``activation_mode`` switches the epilogue.  Those values
        come from the unit tests, so they are declared in the spec and the gate
        has to accept them (they are backed by real arguments) and keep them in
        the reported axis set.
        """

        module = _load_tool("stable_coverage.py")
        report = module.evaluate()
        row = next(entry for entry in report["rows"]
                   if entry["op"] == "npu_causal_conv1d")
        self.assertEqual(row["problems"], [])
        self.assertEqual(row["axes"]["run_mode"], [0, 1])
        self.assertEqual(row["axes"]["head_num"], [0, 2])
        self.assertEqual(row["axes"]["activation_mode"], [0, 1])


class BuildStampTest(unittest.TestCase):
    """The launcher and the glue must come from the same codegen run."""

    def _load_stable_module(self, root: Path) -> dict:
        """Import a throwaway copy of the package so the real one stays out.

        ``_stable`` imports torch lazily, so a private copy is importable
        without torch and without an NPU -- which is the point: the stamp check
        runs before any of that.
        """

        package = root / "fla_npu" / "ops" / "ascendc"
        package.mkdir(parents=True, exist_ok=True)
        for parent in (root / "fla_npu", root / "fla_npu" / "ops", package):
            (parent / "__init__.py").write_text("", encoding="utf-8")
        for name in ("_stable.py", "_stable_generated.py"):
            shutil.copy2(OPS_DIR / name, package / name)
        saved = {name: module_ for name, module_ in sys.modules.items()
                 if name.startswith("fla_npu")}
        for name in list(saved):
            del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("fla_npu.ops.ascendc._stable")
        except Exception:
            sys.path.remove(str(root))
            sys.modules.update(saved)
            raise

        def restore() -> None:
            # The copy has to stay importable while the test runs: the code
            # under test does `from . import _stable_generated`, and tearing the
            # package down early would make it import the real one (or fail on
            # a host without torch) and silently skip the check.
            sys.path.remove(str(root))
            for name in list(sys.modules):
                if name.startswith("fla_npu"):
                    del sys.modules[name]
            sys.modules.update(saved)

        self.addCleanup(restore)
        return module

    def test_mismatched_library_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module = self._load_stable_module(Path(temp_dir))

            class FakeLib:
                @staticmethod
                def fla_npu_thin_source_hash() -> bytes:
                    return b"00000000000000000000000000000000"

            with mock.patch.object(ctypes, "CDLL", return_value=FakeLib()):
                with self.assertRaises(RuntimeError) as caught:
                    module._check_build_stamp("/tmp/libfla_npu_thin.so")
            message = str(caught.exception)
            self.assertIn("different generated adapters", message)
            self.assertIn("build_stable.py", message)

    def test_matching_library_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module = self._load_stable_module(Path(temp_dir))
            expected = module._stable_generated._GENERATED_HASH
            calls = []

            class FakeLib:
                @staticmethod
                def fla_npu_thin_source_hash() -> bytes:
                    calls.append(1)
                    return expected.encode()

            with mock.patch.object(ctypes, "CDLL", return_value=FakeLib()) as cdll:
                module._check_build_stamp("/tmp/libfla_npu_thin.so")
            # Without this the test would pass even if the check bailed out
            # before ever reading the library.
            self.assertTrue(cdll.called, "the stamp check never opened the library")
            self.assertEqual(len(calls), 1, "the stamp was not read exactly once")

    def test_library_without_the_stamp_is_accepted(self) -> None:
        """Artifacts built before the stamp existed must keep working."""

        with tempfile.TemporaryDirectory() as temp_dir:
            module = self._load_stable_module(Path(temp_dir))
            calls = []

            class FakeLib:
                @staticmethod
                def fla_npu_thin_source_hash() -> bytes:
                    calls.append(1)
                    return b"unknown"

            with mock.patch.object(ctypes, "CDLL", return_value=FakeLib()):
                module._check_build_stamp("/tmp/libfla_npu_thin.so")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
