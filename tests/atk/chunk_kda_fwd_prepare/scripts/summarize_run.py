"""Summarize ALL native report rows, retaining failures and missing evidence."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import math
import re
from pathlib import Path

from delivery_cases import OP, attr, load, write_json

MASK_COUNTS = {"none": 6, "recompute": 12, "save": 13, "forward": 7}


def summarize(run):
    run = Path(run)
    identity = load(run / "run.json")
    cases = load(run / f"atk_{OP}.json")
    reports = sorted((run / "results").glob("atk_output/*/report/*.json"))
    records = [row for path in reports for row in load(path)]
    by_id = {}
    for row in records:
        by_id.setdefault(int(row["编号"]), []).append(row)
    accuracy = identity["scope"] in {"accuracy", "model-accuracy"}
    console = run / "console.log"
    dut_errors = set()
    traced_keys = {}
    active_case = None
    if console.exists():
        with console.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                role = re.search(r"KDA_PREPARE_ROLE case=(\d+) backend=(?:npu|pyaclnn) role=dut", line)
                if role:
                    active_case = int(role.group(1))
                key = re.search(r"ChunkKdaFwdPrepare tiling:.*tilingKey=(\d+)", line)
                if key and active_case is not None:
                    traced_keys.setdefault(active_case, Counter())[int(key.group(1))] += 1
                match = re.search(r"KDA_PREPARE_DUT_ERROR case=(\d+)", line)
                if match:
                    dut_errors.add(int(match.group(1)))
    results = []
    for case in cases:
        found = by_id.get(case["id"], [])
        result = dict(id=case["id"], name=case["name"], seed=case["default_seed"],
                      layout=attr(case, "layout"), mode=attr(case, "backward_mode"),
                      status="pending", outputs_compared=0, outputs_passed=0)
        if len(found) > 1:
            result.update(status="duplicate_report", error="Multiple native rows for one ID")
        elif found:
            row = found[0]
            result["native_execution_status"] = row.get("运行结果")
            result["dut_error_logged"] = case["id"] in dut_errors
            reason = row.get("失败原因")
            if row.get("运行结果") != "SUCCESS":
                result.update(status="execution_failure", error=str(reason))
            elif reason and str(reason).strip().lower() not in {"none", "nan", "-", "无"}:
                result.update(status="execution_failure", error=str(reason))
            elif result["dut_error_logged"]:
                result.update(status="execution_failure", error="DUT exception recorded in console.log")
            elif accuracy:
                try:
                    details = row.get("精度详情", {})
                    if isinstance(details, str):
                        details = ast.literal_eval(details)
                    compared = details.get("cpu_cpu_reference", [])
                    names = [item.get("filename") for item in compared]
                    expected = {f"output_{i}.pt" for i in range(MASK_COUNTS[result["mode"]])}
                    failed = [item for item in compared if item.get("result") is not True]
                    result.update(outputs_compared=len(compared),
                                  outputs_passed=sum(item.get("result") is True for item in compared))
                    if len(names) != len(expected) or set(names) != expected:
                        result.update(status="missing_or_duplicate_comparison", expected_outputs=len(expected))
                    elif failed:
                        result.update(status="precision_failure", failed_outputs=failed)
                    else:
                        result.update(status="passed")
                except (ValueError, SyntaxError, TypeError, AttributeError) as exc:
                    result.update(status="report_parse_failure", error=str(exc))
            elif identity["scope"] == "determinism":
                # ATK 26.9.24 exports the unprefixed column; other releases
                # retain the backend prefix. Require every present DC flag to
                # be explicitly true, so conflicting/missing flags fail closed.
                fields = ("确定性计算结果", "npu_npu_dut_确定性计算结果")
                result["native_dc_fields"] = {name: row[name] for name in fields if name in row}
                result["native_dc_result"] = bool(result["native_dc_fields"]) and all(
                    value is True for value in result["native_dc_fields"].values())
                result["status"] = "passed" if result["native_dc_result"] else "determinism_failure_or_missing_evidence"
            elif identity["scope"] == "mssanitizer":
                result["status"] = "passed" if row.get("npu_npu_dut_内存检测通过") is True else "mss_failure_or_missing_evidence"
            elif identity["scope"] == "route-trace":
                keys = traced_keys.get(case["id"], {})
                result["observed_numeric_key_counts"] = dict(keys)
                result["status"] = "traced" if len(keys) == 1 else "trace_missing_or_ambiguous"
            else:
                duration = row.get("npu_npu_dut_Device性能（us）")
                result["device_us"] = duration
                result["status"] = "measured" if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0 else "missing_performance"
        results.append(result)
    statuses = dict(Counter(item["status"] for item in results))
    complete = len(records) == len(cases) and len(by_id) == len(cases) and not statuses.get("pending")
    # DC/MSS still require observed-key and instrumentation evidence. This
    # summary cannot turn release-package runs into an instrumented acceptance.
    all_passed = complete and statuses.get("passed", 0) == len(cases)
    result = dict(scope=identity["scope"], expected_cases=len(cases), reported_rows=len(records),
                  complete=complete, status_counts=statuses, native_all_passed=all_passed,
                  outputs_compared=sum(r["outputs_compared"] for r in results),
                  outputs_passed=sum(r["outputs_passed"] for r in results),
                  observed_numeric_keys=sorted({key for keys in traced_keys.values() for key in keys}),
                  groups_by_layout={layout: dict(Counter(r["status"] for r in results if r["layout"] == layout))
                                    for layout in sorted({r["layout"] for r in results})},
                  identity=identity, reports=[str(p.relative_to(run)) for p in reports], cases=results)
    write_json(run / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    print(json.dumps({k: result[k] for k in ("scope", "expected_cases", "reported_rows", "complete",
          "status_counts", "outputs_compared", "outputs_passed", "groups_by_layout")}, ensure_ascii=False))
