"""Tests for the Stable-ABI gates themselves.

A gate that quietly stops working is worse than no gate: it still prints OK
while nothing is checked.  These tests are offline (no torch, no NPU) and cover
both directions -- each gate passes on the tree as it stands, and it fails on an
input that is deliberately wrong.

Usage:  python -m unittest tests.test_stable_gates
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = REPO_ROOT / "torch_custom" / "fla_npu"
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
SRC_DIR = SETUP_DIR / "csrc_stable" / "src"


def _load_tool(name: str):
    path = SETUP_DIR / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class BuildStampTest(unittest.TestCase):
    """The library stamp and the Python side must be produced together."""

    def test_build_stamp_covers_every_adapter_source(self) -> None:
        import importlib.util as util

        path = SETUP_DIR / "csrc_stable" / "build_stable.py"
        spec = util.spec_from_file_location("build_stable", path)
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        first = module.source_hash()
        sources = sorted((SETUP_DIR / "csrc_stable" / "src").glob("*.cpp"))
        self.assertTrue(sources, "no adapter sources found")
        with tempfile.TemporaryDirectory() as tmp:
            target = sources[0]
            original = target.read_text(encoding="utf-8")
            try:
                target.write_text(original + "\n// changed\n", encoding="utf-8")
                self.assertNotEqual(
                    module.source_hash(), first,
                    "editing one adapter must change the stamp")
            finally:
                target.write_text(original, encoding="utf-8")
        self.assertEqual(module.source_hash(), first)

    def test_checked_in_hash_module_matches_the_sources(self) -> None:
        """A stale _stable_hash.py would reject a freshly built library."""

        import importlib.util as util

        path = SETUP_DIR / "csrc_stable" / "build_stable.py"
        spec = util.spec_from_file_location("build_stable2", path)
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        hash_module = OPS_DIR / "_stable_hash.py"
        if not hash_module.is_file():
            self.skipTest("no _stable_hash.py in this tree (never built here)")
        match = re.search(r'SOURCE_HASH = "([0-9a-f]{32})"',
                          hash_module.read_text(encoding="utf-8"))
        self.assertIsNotNone(match, "_stable_hash.py carries no SOURCE_HASH")
        self.assertEqual(match.group(1), module.source_hash(),
                         "the checked-in stamp does not match the adapters; "
                         "rebuild with python csrc_stable/build_stable.py")


class CoverageGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool("stable_coverage.py")

    def test_current_tree_has_no_unexplained_gap(self) -> None:
        report = self.tool.evaluate()
        self.assertEqual(report["blockers"], [])
        self.assertGreater(report["adapter_count"], 25)

    def test_missing_wrapper_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ops = Path(tmp) / "ops"
            ops.mkdir()
            original = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
            # Drop one wrapper: the operator must then be reported, not skipped.
            stripped = re.sub(
                r"^def npu_chunk_fwd_o\(.*?(?=^def )", "", original,
                flags=re.S | re.M)
            self.assertNotEqual(stripped, original)
            (ops / "_stable.py").write_text(stripped, encoding="utf-8")
            (ops / "__init__.py").write_text(
                (OPS_DIR / "__init__.py").read_text(encoding="utf-8"),
                encoding="utf-8")
            (ops / "_aclnn_ctypes.py").write_text(
                (OPS_DIR / "_aclnn_ctypes.py").read_text(encoding="utf-8"),
                encoding="utf-8")
            with mock.patch.object(self.tool, "OPS_DIR", ops):
                report = self.tool.evaluate()
            self.assertTrue(any("npu_chunk_fwd_o: no wrapper" in item
                                for item in report["blockers"]),
                            report["blockers"])

    def test_enum_table_order_drift_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            target = src / "stable_gdn.cpp"
            text = target.read_text(encoding="utf-8")
            # Swap two layout names in the C++ table only: the Python table no
            # longer agrees, which is exactly the silent-layout-change bug.
            changed = text.replace(
                'kGdnFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"}',
                'kGdnFwdLayoutNames[] = {"BNSD", "BSND", "TND", "NTD"}')
            self.assertNotEqual(changed, text)
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("kGdnFwdLayoutNames order" in item
                                for item in report["blockers"]),
                            report["blockers"])


class AbiParityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool("op_abi_parity.py")

    def test_current_tree_matches(self) -> None:
        report = self.tool.evaluate()
        self.assertEqual(report["problems"], [])
        self.assertGreater(report["checked"], 25)

    def test_parameter_reorder_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            for path in SRC_DIR.glob("stable_*.cpp"):
                (src / path.name).write_text(path.read_text(encoding="utf-8"),
                                             encoding="utf-8")
            target = src / "stable_kda.cpp"
            text = target.read_text(encoding="utf-8")
            changed = text.replace(
                "run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,\n"
                "                               std::optional<Tensor> dt_bias,\n"
                "                               std::optional<Tensor> cu_seqlens,\n"
                "                               int64_t chunk_size,",
                "run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,\n"
                "                               std::optional<Tensor> dt_bias,\n"
                "                               std::optional<Tensor> cu_seqlens,\n"
                "                               bool chunk_size,")
            self.assertNotEqual(changed, text, "test setup did not apply")
            target.write_text(changed, encoding="utf-8")
            with mock.patch.object(self.tool, "SRC_DIR", src):
                report = self.tool.evaluate()
            self.assertTrue(any("chunk_size" in item for item in report["problems"]),
                            report["problems"])


class FallbackGateTest(unittest.TestCase):
    def test_no_adapter_reaches_the_ctypes_reference(self) -> None:
        tool = _load_tool("stable_ctypes_fallbacks.py")
        text = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
        self.assertEqual(tool.delegating_ops(text), [],
                         "an adapter delegates to ctypes again")

    def test_delegation_is_detected(self) -> None:
        tool = _load_tool("stable_ctypes_fallbacks.py")
        sample = ("def npu_x(a):\n"
                  "    from . import _aclnn_ctypes as ct\n"
                  "    return ct.npu_x(a)\n")
        self.assertEqual(tool.delegating_ops(sample), ["npu_x"])


class CtypesTableGateTest(unittest.TestCase):
    """The ctypes argument table is what the OPP headers are compared against."""

    def test_every_entry_ends_with_workspace_and_executor(self) -> None:
        tool = _load_tool("op_abi_validate.py")
        table = tool.parse_ctypes_table(OPS_DIR / "_aclnn_ctypes.py")
        self.assertGreaterEqual(len(table), 20)
        for symbol, kinds in table.items():
            with self.subTest(symbol=symbol):
                # The trailing pair is dropped by the parser, so what is left
                # must not contain a pointer-to-out-parameter.
                self.assertNotIn("_pointer", kinds)
                self.assertTrue(kinds, f"{symbol} parsed to nothing")

    def test_header_kinds_are_recognised(self) -> None:
        tool = _load_tool("op_abi_validate.py")
        self.assertEqual(tool.header_kind("const aclTensor *q"), tool.WILDCARD)
        self.assertEqual(tool.header_kind("const aclIntArray *cu"), tool.WILDCARD)
        self.assertEqual(tool.header_kind("int64_t chunkSize"), "int64")
        self.assertEqual(tool.header_kind("bool useExp2"), "bool")
        self.assertEqual(tool.header_kind("double scale"), "double")
        self.assertEqual(tool.header_kind("const char *layout"), "char_ptr")


if __name__ == "__main__":
    unittest.main()
