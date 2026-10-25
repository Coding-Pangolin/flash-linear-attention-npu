"""Guard against real failure modes: late warnings and native success without a trace."""
from pathlib import Path
import tempfile
import unittest

from audit_mss import inspect_log


class RawMssAuditTest(unittest.TestCase):
    def test_warning_is_not_an_error_and_duplicate_launch_is_incomplete(self):
        start = "Start memcheck sanitizer on kernel ChunkKdaFwdPrepare_abc_2570\n"
        finish = "Sanitizer finished on kernel ChunkKdaFwdPrepare_abc_2570. See all detected errors above.\n"
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"sanitizer.log"
            log.write_text(start+"====== WARNING: out of bounds of size 128 on GM when writing data\n"+finish)
            result = inspect_log(log, [2570], "memcheck")
            self.assertTrue(result["complete_instrumentation"])
            self.assertEqual(result["error_lines"], 0)
            self.assertEqual(result["warning_classes"], {"gm_ownership": 1})
            self.assertEqual(result["review_state"], "warning_review_required")
            self.assertFalse(result["raw_log_clean"])
            log.write_text(start+"====== WARNING: out of bounds of size 128\n"
                           "====== at 0x1234 on GM when writing data\n"+finish)
            self.assertEqual(inspect_log(log, [2570], "memcheck")["warning_classes"],
                             {"gm_ownership": 1})
            log.write_text((start+finish)*2)
            self.assertFalse(inspect_log(log, [2570], "memcheck")["complete_instrumentation"])

    def test_target_error_remains_an_unclosed_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"sanitizer.log"
            log.write_text("Start racecheck sanitizer on kernel ChunkKdaFwdPrepare_abc_68126\n"
                           "====== ERROR: potential RAW race on UB\n"
                           "Sanitizer finished on kernel ChunkKdaFwdPrepare_abc_68126. See all detected errors above.\n")
            result = inspect_log(log, [68126], "racecheck")
            self.assertEqual(result["errors_by_key"], {"68126": 1})
            self.assertEqual(result["review_state"], "error_review_required")
            self.assertFalse(result["raw_log_clean"])

    def test_missing_finish_wrong_key_tool_and_late_warning_cannot_pass(self):
        start = "[mssanitizer] Start memcheck sanitizer on kernel ChunkKdaFwdPrepare_abc_2570\n"
        finish = "[mssanitizer] Sanitizer finished on kernel ChunkKdaFwdPrepare_abc_2570. No error detected.\n"
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"sanitizer.log"
            log.write_text(start+finish)
            self.assertTrue(inspect_log(log, [2570], "memcheck")["raw_log_clean"])
            self.assertFalse(inspect_log(log, [2570, 2590], "memcheck")["raw_log_clean"])
            self.assertFalse(inspect_log(log, [2570], "racecheck")["raw_log_clean"])
            for text in ("", start, start+finish+"====== WARNING: out of bounds of size 128\n",
                         start+finish+"====== ERROR: GM read before initialization\n",
                         start+finish.replace("No error detected.", "See all detected errors above.")):
                log.write_text(text)
                self.assertFalse(inspect_log(log, [2570], "memcheck")["raw_log_clean"])


if __name__ == "__main__":
    unittest.main()
