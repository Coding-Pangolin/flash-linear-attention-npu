"""Stdlib tests plus optional torch reference tests (no device allocation)."""
import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from delivery_cases import ROOT, OP, SEEDS, attr, expand_seeds, inputs, load, validate
from run_delivery import check_accuracy
from summarize_run import summarize


class ManifestTests(unittest.TestCase):
    def test_default_cli_keeps_original_seeds_and_case_counts(self):
        # Exercise the public CLI: a past runner silently tripled model cases.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "unit_only_identity"
            artifact.write_bytes(b"dry-run identity; not an executable package")
            for scope, count, suffix in (("accuracy", 200, ""), ("model-accuracy", 128, "_perf")):
                out = root / scope
                subprocess.run([sys.executable, "-B", str(ROOT/"scripts/run_delivery.py"),
                    "--scope", scope, "--out", str(out), "--device", "0",
                    "--source-commit", "unit-dry-run", "--package", str(artifact),
                    "--kernel-lib", str(artifact), "--dry-run"], check=True, capture_output=True)
                actual = load(out/f"atk_{OP}.json")
                original = load(ROOT/f"atk_{OP}{suffix}.json")
                self.assertEqual(len(actual), count)
                self.assertEqual([(c["id"], c["default_seed"]) for c in actual],
                                 [(c["id"], c["default_seed"]) for c in original])
                self.assertEqual(load(out/"run.json")["acceptance"], "not_run")

    def test_attrs_null_keeps_native_list_container(self):
        case = load(ROOT / f"atk_{OP}_paths.json")[0]
        inputs(case)
        for name in ("cu_seqlens", "chunk_indices"):
            broken = copy.deepcopy(case)
            index = next(i for i, value in enumerate(broken["inputs"])
                         if isinstance(value, list) and value[0]["name"] == name)
            broken["inputs"][index] = broken["inputs"][index][0]
            with self.assertRaisesRegex(ValueError, "requires a list"):
                inputs(broken)

    def test_full_template_and_interface_coverage(self):
        self.assertEqual(validate()["mss_template_cases"], 6)

    def test_seed_expansion_preserves_every_input(self):
        cases = load(ROOT / f"atk_{OP}.json")
        expanded = expand_seeds(cases)
        self.assertEqual(len(expanded), 600)
        for index, case in enumerate(cases):
            group = expanded[index*3:index*3+3]
            self.assertEqual([c["default_seed"] for c in group], list(SEEDS))
            for item in group:
                self.assertEqual(case["inputs"], item["inputs"])

    def test_report_checker_rejects_false_missing_duplicate_and_wrong_mask(self):
        cases = load(ROOT / f"atk_{OP}.json")[:1]
        n = {"none": 6, "forward": 7, "save": 13, "recompute": 12}[attr(cases[0], "backward_mode")]
        good = {"编号": cases[0]["id"], "失败原因": "", "运行结果": "SUCCESS",
                "精度详情": str({"cpu_cpu_reference": [
                    {"filename": f"output_{i}.pt", "result": True} for i in range(n)]})}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"report.json"
            path.write_text(json.dumps([good]), encoding="utf-8")
            self.assertEqual(check_accuracy([path], cases)["passed"], 1)
            console = Path(folder)/"console.log"
            console.write_text(f"KDA_PREPARE_DUT_ERROR case={cases[0]['id']}\n")
            with self.assertRaisesRegex(RuntimeError, "DUT exception"):
                check_accuracy([path], cases, console)
            for rows in ([], [good, good], [dict(good, 精度详情=str({
                    "cpu_cpu_reference": [{"result": False}]*n}))],
                    [dict(good, 精度详情=str({"cpu_cpu_reference": []}))],
                    [dict(good, 精度详情=str({"cpu_cpu_reference": [
                        {"filename": "output_0.pt", "result": True}]*n}))],
                    [dict(good, 失败原因="DUT failed")],
                    [dict(good, 运行结果="FAILED")]):
                path.write_text(json.dumps(rows), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    check_accuracy([path], cases)

    def test_sources_compile_without_bytecode(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_dc_summary_uses_native_dc_result_not_accuracy_result(self):
        case = load(ROOT / f"atk_{OP}.json")[0]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/"run.json").write_text(json.dumps({"scope": "determinism"}))
            (run/f"atk_{OP}.json").write_text(json.dumps([case]))
            report = run/"results/atk_output/native/report/result.json"
            report.parent.mkdir(parents=True)
            row = {"编号": case["id"], "运行结果": "SUCCESS", "失败原因": "",
                   "npu_npu_dut_精度通过": False, "npu_npu_dut_确定性计算结果": True}
            report.write_text(json.dumps([row]))
            self.assertTrue(summarize(run)["native_all_passed"])
            row.update(npu_npu_dut_精度通过=True, npu_npu_dut_确定性计算结果=False)
            report.write_text(json.dumps([row]))
            self.assertFalse(summarize(run)["native_all_passed"])
            row["npu_npu_dut_确定性计算结果"] = True
            report.write_text(json.dumps([row]))
            # Actual ATK 26.9.24 reports the unprefixed name. Accuracy flags
            # never substitute for either schema's native determinism flag.
            del row["npu_npu_dut_确定性计算结果"]
            for value, expected in ((True, True), (False, False), (None, False)):
                row["确定性计算结果"] = value
                report.write_text(json.dumps([row]))
                self.assertEqual(summarize(run)["native_all_passed"], expected)
            del row["确定性计算结果"]
            report.write_text(json.dumps([row]))
            self.assertFalse(summarize(run)["native_all_passed"])
            row.update(确定性计算结果=True, npu_npu_dut_确定性计算结果=False)
            report.write_text(json.dumps([row]))
            self.assertFalse(summarize(run)["native_all_passed"])
            row["npu_npu_dut_确定性计算结果"] = True
            report.write_text(json.dumps([row]))
            (run/"console.log").write_text(f"KDA_PREPARE_DUT_ERROR case={case['id']}\n")
            self.assertFalse(summarize(run)["native_all_passed"])

    def test_models_preserve_all_boundaries(self):
        manifest = load(ROOT / "model_cases/manifest.json")
        cases = load(ROOT / f"atk_{OP}_perf.json")
        self.assertEqual(len(cases), 128)
        for index, case in enumerate(cases):
            model = manifest["cases"][index % 32]
            actual = case["inputs"][19-1]  # cu_seqlens: slot 18
            self.assertEqual([x["range_values"] for x in actual],
                             manifest["groups"][model["group"]])


try:
    import torch
    sys.path.insert(0, str(ROOT))
    import executor_chunk_kda_fwd_prepare as ref
except ModuleNotFoundError as exc:
    if exc.name not in {"torch", "atk"}:
        raise
    torch = None


@unittest.skipIf(torch is None, "requires torch and installed ATK")
class ReferenceTests(unittest.TestCase):
    def values(self, layout="BNSD", grouped=False):
        torch.set_num_threads(1)
        h, hv, t = 1, 2 if grouped else 1, 3
        q = torch.zeros(1, h, t, 128, dtype=torch.bfloat16)
        v = torch.arange(hv*t*128).reshape(1, hv, t, 128).float().div(100).bfloat16()
        g = torch.zeros(1, hv, t, 128)
        beta = torch.full((1, hv, t), .5, dtype=torch.bfloat16)
        def convert(x):
            if layout == "BNSD": return x
            if layout == "BSND": return x.transpose(1, 2).contiguous()
            if layout == "TND": return x.transpose(1, 2).squeeze(0).contiguous()
            return x.squeeze(0)
        return dict(q=convert(q), k=convert(q.clone()), v=convert(v), g=convert(g),
            beta=convert(beta), scale=128**-.5, layout=layout, chunk_size=64,
            epsilon=1e-6, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
            safe_gate=False, lower_bound=-5., use_exp2=True, a_log=None,
            dt_bias=None, cu_seqlens=[0, 1, 3] if layout in {"TND", "NTD"} else None,
            chunk_indices=None, backward_mode="save")

    def test_closed_form_zero_qk_all_layouts_modes_and_gva(self):
        for layout in ("BNSD", "BSND", "TND", "NTD"):
            values = self.values(layout, grouped=True)
            spec = ref._derive_spec(values, 0)
            prepared = ref._direct_inputs(values, high_precision=False)
            for mode in ("none", "recompute", "save", "forward"):
                spec["backward_mode"] = mode
                for role in ("truth_fp64", "control_fp32_bf16"):
                    out = ref.run_cpu(spec, prepared, role)
                    ref._validate_output_contract(spec, out, check_dtype=role.startswith("control"))
                    self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
                    self.assertTrue(torch.equal(out[3], torch.zeros_like(out[3])))
                    # L=0 => A=I, therefore u=beta*v for all tokens and heads.
                    expected = ref._layout_to_bsnd(values["v"], layout).permute(0, 2, 1, 3).double()*.5
                    if layout in {"TND", "NTD"}: expected = expected.squeeze(0)
                    if role.startswith("control"): expected = expected.bfloat16()
                    torch.testing.assert_close(out[4], expected, rtol=0, atol=0, check_dtype=False)

    def test_inverse_is_exact_not_truncated_neumann(self):
        torch.manual_seed(3)
        l = torch.tril(torch.randn(2, 64, 64, dtype=torch.float64)*.05, diagonal=-1)
        actual = ref._block_inverse_batched(l, lambda value, dtype: value.to(dtype))
        eye = torch.eye(64, dtype=torch.float64).expand(2, -1, -1)
        torch.testing.assert_close((eye+l)@actual, eye, rtol=1e-12, atol=1e-12)

    def test_roles_keep_distinct_precision(self):
        values = self.values()
        values["q"].fill_(.03)
        values["k"].fill_(.03)
        spec = ref._derive_spec(values, 0)
        prepared = ref._direct_inputs(values, high_precision=False)
        high = ref.run_cpu(spec, prepared, "truth_fp64")
        low = ref.run_cpu(spec, prepared, "control_fp32_bf16")
        self.assertTrue(all(x.dtype == torch.float64 for x in high))
        self.assertEqual([x.dtype for x in low], list(ref.OUTPUT_DTYPES))
        self.assertTrue(any(not torch.equal(a.float(), b.float()) for a, b in zip(high, low)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
