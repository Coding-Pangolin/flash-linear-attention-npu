"""Offline checks for real model boundaries, ATK encoding and acceptance gates."""

import copy
import tempfile
import unittest
from pathlib import Path

from prepare_model_cases import OP_DIR, PUBLIC_INPUTS, load, model_cases, prepare, public_inputs
from run_model_cases import accuracy_verdict, command_for, validate_plan


class ModelCaseTests(unittest.TestCase):
    def test_boundaries_and_optional_heads(self):
        manifest = load(OP_DIR / "model_cases/manifest.json")
        self.assertEqual(len(manifest["groups"]), 16)
        self.assertEqual(len(model_cases(manifest)), 32)
        optional = model_cases(manifest, True)
        self.assertEqual(len(optional), 48)
        for case in optional:
            cu = manifest["groups"][case["group"]]
            self.assertEqual(cu[-1], case["T"])
            self.assertEqual(len(cu)-1, case["sequences"])
        invalid = copy.deepcopy(manifest)
        invalid["groups"]["V01"][-1] -= 1
        with self.assertRaises(ValueError):
            model_cases(invalid)

    def test_complete_matrix_and_public_host_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "generated"
            plan = prepare(output, OP_DIR / "model_cases/profiles.json")
            self.assertEqual((plan["accuracy_cases"], plan["performance_cases"]), (128, 128))
            self.assertEqual(validate_plan(output), plan)
            manifest = load(OP_DIR / "model_cases/manifest.json")
            templates = {c["id"]:c for c in load(OP_DIR / "atk_chunk_kda_fwd_prepare.json")}
            all_names, all_ids = [], []
            for suite in plan["suites"]:
                records = load(output / suite["files"]["accuracy"]["file"])
                self.assertEqual(len(records), 32)
                for case in records:
                    inputs = public_inputs(case)
                    self.assertEqual(len(inputs), len(PUBLIC_INPUTS))
                    all_names.append(case["name"])
                    all_ids.append(case["id"])
                    model_id = int(case["name"].split("_")[2][1:])
                    model = manifest["cases"][model_id-1]
                    cu_items = case["inputs"][PUBLIC_INPUTS.index("cu_seqlens")]
                    self.assertEqual([i["range_values"] for i in cu_items], manifest["groups"][model["group"]])
                    self.assertTrue(all(i["dtype"] == "int" and i["type"] == "attrs" for i in cu_items))
                    self.assertEqual(inputs["chunk_indices"]["range_values"], "null")
                    self.assertEqual(inputs["backward_mode"]["range_values"], "save")
                    self.assertEqual(inputs["q"]["shape"], [1,model["H"],model["T"],128])
                    self.assertEqual(inputs["layout"]["range_values"], "BNSD")
                    self.assertIn("cv_fused_double_benchmark", case["standard"]["acc"])
                    old_inputs = public_inputs(templates[suite["profile"]["template_id"]])
                    for name in ("q", "k", "v", "g", "beta", "a_log", "dt_bias"):
                        self.assertEqual(inputs[name]["range_values"], old_inputs[name]["range_values"])
                for case in load(output / suite["files"]["performance"]["file"]):
                    self.assertEqual(case["default_seed"], 3)
            self.assertEqual(len(set(all_names)), 128)
            self.assertEqual(len(set(all_ids)), 128)
            changed = output / plan["suites"][0]["files"]["accuracy"]["file"]
            changed.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_plan(output)

    def test_h2_and_command_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = prepare(Path(temp) / "all_heads", OP_DIR / "model_cases/profiles.json", True)
            self.assertEqual((plan["accuracy_cases"], plan["performance_cases"]), (192, 192))
        accuracy = command_for("atk", "accuracy", Path("a.json"), Path("out"), 7, 100)
        perf = command_for("atk", "performance", Path("p.json"), Path("out"), 0, 100)
        self.assertIn("cpu_golden", accuracy)
        self.assertIn("--gm_init_flag", accuracy)
        self.assertNotIn("cpu_golden", perf)
        self.assertIn("performance_device", perf)
        self.assertNotIn("--bm_device", perf)

    def test_report_gate_rejects_missing_partial_and_failed(self):
        import openpyxl

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.assertEqual(accuracy_verdict(folder, 96)["status"], "NOT_VERIFIED")
            path = folder / "report.xlsx"
            for counts, label, expected in [((96,96,0,96),"Passed","PASS"),
                                             ((95,95,0,95),"Passed","NOT_VERIFIED"),
                                             ((96,96,0,95),"Failed","NOT_VERIFIED"),
                                             ((96,96,0,96),"-","NOT_VERIFIED")]:
                book = openpyxl.Workbook()
                sheet = book.active
                sheet.title = "summary"
                sheet.append(["节点", "总用例数", "执行成功用例个数", "执行失败用例个数", "通过用例个数", "精度是否达标"])
                sheet.append(["npu_dut", *counts, label])
                book.save(path)
                book.close()
                self.assertEqual(accuracy_verdict(folder, 96)["status"], expected)


if __name__ == "__main__":
    unittest.main()
