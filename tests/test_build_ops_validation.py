#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for FLA_NPU_OPS/--ops 前置校验 (issue #482)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CHECKER = SCRIPTS_DIR / "check_build_ops.py"

sys.path.insert(0, str(SCRIPTS_DIR))
from check_build_ops import (  # noqa: E402
    OP_SEARCH_ROOTS,
    discover_supported_ops,
    format_validation_error,
    is_build_all,
    parse_ops_filter,
    unsupported_ops,
    validate_ops_filter,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class SupportedOperatorDiscoveryTest(unittest.TestCase):
    def test_operator_list_comes_from_repository_layout(self) -> None:
        ops = discover_supported_ops(REPO_ROOT)

        self.assertTrue(ops, "repository should expose at least one operator")
        self.assertEqual(ops, sorted(ops))
        self.assertEqual(len(ops), len(set(ops)), "operator names must be unique")
        for expected in ("recompute_w_u_fwd", "chunk_gated_delta_rule_fwd"):
            self.assertIn(expected, ops)

    def test_every_listed_operator_exists_in_the_source_tree(self) -> None:
        components = set()
        for search_root in OP_SEARCH_ROOTS:
            base = REPO_ROOT / search_root
            if not base.is_dir():
                continue
            for path in base.rglob("CMakeLists.txt"):
                components.update(path.parts)
        self.assertTrue(set(discover_supported_ops(REPO_ROOT)).issubset(components))

    def test_discovery_mirrors_cmake_layout_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "fla/ops/ascendc/gdn/chunk_gdn_fwd/op_name_a/op_host/CMakeLists.txt", "#\n")
            _write(root / "fla/ops/ascendc/kda/op_name_b/CMakeLists.txt", "#\n")
            _write(root / "fla/ops/ascendc/kda/op_name_b/op_host/tests/CMakeLists.txt", "#\n")
            _write(root / "fla/ops/triton/triton_core/CMakeLists.txt", "#\n")

            self.assertEqual(discover_supported_ops(root), ["op_name_a", "op_name_b"])


class OpsFilterParsingTest(unittest.TestCase):
    def test_value_is_split_on_commas_and_stripped(self) -> None:
        self.assertEqual(
            parse_ops_filter(" recompute_w_u_fwd , chunk_fwd_h "),
            ["recompute_w_u_fwd", "chunk_fwd_h"],
        )

    def test_empty_entries_are_dropped(self) -> None:
        self.assertEqual(parse_ops_filter("recompute_w_u_fwd,,"), ["recompute_w_u_fwd"])

    def test_unset_and_empty_values_mean_full_build(self) -> None:
        for value in (None, "", "   ", ",", ", ,"):
            self.assertEqual(parse_ops_filter(value), [])
            self.assertTrue(is_build_all(value))
            self.assertEqual(unsupported_ops(value), [])

    def test_all_is_a_full_build_sentinel(self) -> None:
        self.assertTrue(is_build_all("all"))
        self.assertTrue(is_build_all("ALL"))
        self.assertEqual(unsupported_ops("all"), [])

    def test_explicit_operator_is_not_a_full_build(self) -> None:
        self.assertFalse(is_build_all("recompute_w_u_fwd"))


class OpsFilterValidationTest(unittest.TestCase):
    def test_supported_single_operator_build_is_accepted(self) -> None:
        self.assertEqual(unsupported_ops("recompute_w_u_fwd"), [])
        self.assertTrue(validate_ops_filter("recompute_w_u_fwd", stream=io.StringIO()))

    def test_supported_multi_operator_build_is_accepted(self) -> None:
        value = "recompute_w_u_fwd,chunk_fwd_h,chunk_fwd_o"
        self.assertEqual(unsupported_ops(value), [])
        self.assertTrue(validate_ops_filter(value, stream=io.StringIO()))

    def test_single_unknown_operator_is_rejected(self) -> None:
        self.assertEqual(unsupported_ops("not_exist_op"), ["not_exist_op"])
        self.assertFalse(validate_ops_filter("not_exist_op", stream=io.StringIO()))

    def test_multiple_unknown_operators_are_all_reported(self) -> None:
        self.assertEqual(unsupported_ops("not_exist_op,also_missing"), ["not_exist_op", "also_missing"])

    def test_mixed_value_fails_as_a_whole(self) -> None:
        value = "recompute_w_u_fwd,not_exist_op"
        self.assertEqual(unsupported_ops(value), ["not_exist_op"])
        self.assertFalse(validate_ops_filter(value, stream=io.StringIO()))

    def test_duplicate_unknown_names_are_reported_once(self) -> None:
        self.assertEqual(unsupported_ops("not_exist_op,not_exist_op"), ["not_exist_op"])

    def test_error_message_carries_names_source_and_supported_list(self) -> None:
        stream = io.StringIO()
        self.assertFalse(
            validate_ops_filter(
                "not_exist_op",
                source="--ops",
                origin="bash build.sh command line",
                stream=stream,
            )
        )
        message = stream.getvalue()
        self.assertIn("Unsupported operator(s) in --ops: not_exist_op", message)
        self.assertIn("Parameter source: bash build.sh command line", message)
        self.assertIn("recompute_w_u_fwd", message)
        self.assertIn("--list-ops", message)

    def test_supported_list_can_be_injected(self) -> None:
        message = format_validation_error(
            "typo_op", source="--ops", unknown=["typo_op"], supported=["good_op", "other_op"]
        )
        self.assertIn("Supported operators (2): good_op, other_op", message)


class CheckerCliTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(CHECKER), *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

    def test_list_flag_prints_supported_operators(self) -> None:
        result = self._run("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(listed, discover_supported_ops(REPO_ROOT))

    def test_json_flag_prints_machine_readable_list(self) -> None:
        result = self._run("--list", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["count"], len(payload["operators"]))
        self.assertIn("recompute_w_u_fwd", payload["operators"])

    def test_unknown_operator_exits_non_zero(self) -> None:
        result = self._run("--ops=not_exist_op", "--source=--ops")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not_exist_op", result.stderr)

    def test_known_operator_exits_zero(self) -> None:
        result = self._run("--ops=recompute_w_u_fwd")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")


@unittest.skipUnless(shutil.which("bash"), "bash is required to exercise build.sh")
class BuildEntryPointTest(unittest.TestCase):
    def _run_build(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "build.sh", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "FLA_NPU_DISABLE_PTH": "1"},
        )

    def test_build_sh_lists_operators_without_building(self) -> None:
        result = self._run_build("--list-ops")
        self.assertEqual(result.returncode, 0, result.stderr)
        # build.sh pipes its stdout through gawk, which prefixes a timestamp.
        listed = [
            line.split("] ", 1)[-1].strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        self.assertIn("recompute_w_u_fwd", listed)
        self.assertIn("chunk_gated_delta_rule_fwd", listed)
        self.assertEqual(sorted(listed), discover_supported_ops(REPO_ROOT))

    def test_build_sh_rejects_unknown_operator_before_building(self) -> None:
        result = self._run_build("--pkg", "--ops=not_exist_op")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported operator(s) in --ops: not_exist_op", result.stderr)
        self.assertIn("Supported operators", result.stderr)

    def test_build_sh_rejects_unknown_operator_mixed_into_valid_list(self) -> None:
        result = self._run_build("--pkg", "--ops=recompute_w_u_fwd,not_exist_op")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not_exist_op", result.stderr)

    def test_build_sh_runs_validation_before_environment_setup(self) -> None:
        source = (REPO_ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn("check_ops_filter", source)
        self.assertIn("scripts/check_build_ops.py", source)
        self.assertIn("--list-ops", source)
        self.assertLess(
            source.index("check_ops_filter\n"),
            source.index("set_env\n\nclean\nclean_build_out"),
        )

    def test_setup_py_validates_fla_npu_ops(self) -> None:
        source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("from check_build_ops import", source)
        self.assertIn("FLA_NPU_OPS", source)
        self.assertIn("_check_ops_filter(ops_filter)", source)


if __name__ == "__main__":
    unittest.main()