"""Regression tests for Prepare's native ATK acceptance boundary."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import openpyxl

PATH = Path(__file__).with_name('check_atk_result.py')
spec = importlib.util.spec_from_file_location('checker', PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrepareAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xlsx = self.root / 'report.xlsx'
        self.case = {'id': 7, 'expected_error_msg': None,
                     'inputs': [{'name': 'backward_mode', 'type': 'attr', 'range_values': ['save']}]}
        self.row = {'编号': 7, '运行结果': 'SUCCESS', '失败原因': None,
                    '用例json信息': json.dumps(self.case),
                    'output_info_list': [{'shape': [1], 'dtype': 'float32'} for _ in range(13)],
                    '精度详情': {'cpu_cpu_reference': [
                        {'filename': f'output_{i}.pt', 'result': True} for i in range(13)]},
                    'npu_npu_dut_确定性计算结果': True,
                    'npu_npu_dut_内存检测通过': True,
                    'npu_npu_dut_Device性能（us）': 1.25}

    def audit(self, kind='accuracy', rows=None, expected=None):
        self.xlsx.with_suffix('.json').write_text(json.dumps(rows or [self.row]), encoding='utf-8')
        return checker._audit_prepare_rows(str(self.xlsx), kind, expected)

    def test_actual_outputs_required_for_all_modes(self):
        for mode, count in [('none', 6), ('forward', 7), ('recompute', 12), ('save', 13)]:
            with self.subTest(mode=mode):
                row = copy.deepcopy(self.row)
                case = dict(self.case, inputs=[{'name': 'backward_mode', 'type': 'attr', 'range_values': [mode]}])
                row['用例json信息'] = json.dumps(case)
                row['output_info_list'] = row['output_info_list'][:count]
                row['精度详情']['cpu_cpu_reference'] = row['精度详情']['cpu_cpu_reference'][:count]
                self.assertEqual(self.audit(rows=[row], expected=[case]), 1)

    def test_empty_or_whitespace_error_sentinel_rejected(self):
        for sentinel in ['', ' ', '\t\n']:
            with self.subTest(sentinel=repr(sentinel)):
                self.row['用例json信息'] = json.dumps(dict(self.case, expected_error_msg=sentinel))
                with self.assertRaisesRegex(ValueError, 'blank'):
                    self.audit()
                path = self.root / 'cases.json'
                path.write_text(json.dumps([dict(self.case, expected_error_msg=sentinel)]))
                with self.assertRaisesRegex(ValueError, 'nonblank'):
                    checker.validate_case_file(path)

    def test_real_failure_not_hidden_by_success_summary(self):
        for field, value in [('运行结果', 'FAILED'), ('失败原因', 'RuntimeError: injected DUT failure'),
                             ('output_info_list', []), ('精度详情', {})]:
            with self.subTest(field=field):
                row = dict(self.row, **{field: value})
                with self.assertRaises(ValueError):
                    self.audit(rows=[row])

    def test_missing_duplicate_and_failed_output_comparisons(self):
        for mutate in ['missing', 'duplicate', 'failed', 'wrong_count', 'empty_shape']:
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(self.row)
                group = row['精度详情']['cpu_cpu_reference']
                if mutate == 'missing': group.pop()
                elif mutate == 'duplicate': group[0] = group[1]
                elif mutate == 'failed': group[-1]['result'] = False
                elif mutate == 'wrong_count': row['output_info_list'].pop()
                else: row['output_info_list'][0]['shape'] = [0]
                with self.assertRaises(ValueError):
                    self.audit(rows=[row])

    def test_case_coverage_missing_duplicate_or_wrong_id(self):
        with self.assertRaisesRegex(ValueError, 'IDs/count'):
            self.audit(expected=[self.case, dict(self.case, id=8)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.audit(rows=[self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'IDs/count'):
            self.audit(expected=[dict(self.case, id=8)])

    def test_dc_cannot_pass_with_empty_outputs(self):
        self.assertEqual(self.audit('determinism'), 1)
        self.row['output_info_list'] = []
        with self.assertRaisesRegex(ValueError, 'no output'):
            self.audit('determinism')

    def test_same_id_cannot_hide_different_case_inputs(self):
        with self.assertRaisesRegex(ValueError, 'input/seed identity'):
            self.audit(expected=[dict(self.case, default_seed=17)])
        self.row['用例json信息'] = json.dumps(dict(self.case, id=8))
        with self.assertRaisesRegex(ValueError, 'identity disagree'):
            self.audit()

    def test_native_resolved_attributes_and_nested_optional_groups(self):
        manifest = dict(self.case, inputs=[self.case['inputs'][0],
                        {'name': 'cu_seqlens', 'type': 'attrs', 'range_values': [[0, 1, 65]]}])
        resolved = dict(self.case, inputs=[dict(self.case['inputs'][0], range_values='save'),
                        [{'name': 'cu_seqlens', 'type': 'attrs', 'range_values': [0, 1, 65]}]])
        self.row['用例json信息'] = json.dumps(resolved)
        self.assertEqual(self.audit(expected=[manifest]), 1)

    def test_dc_and_mss_require_native_pass_flags(self):
        for kind, field in [('determinism', 'npu_npu_dut_确定性计算结果'),
                            ('mssanitizer', 'npu_npu_dut_内存检测通过')]:
            with self.subTest(kind=kind):
                row = dict(self.row, **{field: False})
                with self.assertRaises(ValueError):
                    self.audit(kind, rows=[row])

    def test_performance_requires_finite_positive_duration(self):
        self.assertEqual(self.audit('performance'), 1)
        for duration in [None, 0, -1, float('nan'), float('inf')]:
            with self.subTest(duration=duration):
                row = dict(self.row, **{'npu_npu_dut_Device性能（us）': duration})
                with self.assertRaises(ValueError):
                    self.audit('performance', rows=[row])

    def test_xlsx_fallback_reads_native_rows(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = 'statistic'
        sheet.append(list(self.row))
        sheet.append([json.dumps(value) if isinstance(value, (dict, list)) else value
                      for value in self.row.values()])
        book.save(self.xlsx)
        self.assertEqual(checker._audit_prepare_rows(str(self.xlsx), 'accuracy'), 1)

    def test_null_and_explicit_negative_patterns_are_not_rewritten(self):
        cases = [self.case, dict(self.case, id=8, expected_error_msg='invalid shape')]
        path = self.root / 'cases.json'
        path.write_text(json.dumps(cases))
        self.assertEqual(checker.validate_case_file(path), cases)

    def test_missing_report_and_empty_summary_fail_closed(self):
        result = subprocess.run([sys.executable, str(PATH), '--type', 'accuracy',
                                 '--op', 'chunk_kda_fwd_prepare_a2', '--output-root', str(self.root)],
                                capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(checker._extract_summary_row(['总用例数', '精度是否达标'], [[0, 'Pass']])['all_pass'])
        self.assertFalse(checker._extract_summary_row(['总用例数', '精度是否达标'], [[1, 'Failed']])['all_pass'])


if __name__ == '__main__':
    unittest.main()
