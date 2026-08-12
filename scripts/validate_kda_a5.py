#!/usr/bin/env python3
"""Run the PR264 A5 acceptance cases and write a compact result bundle."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str


CASES = (
    Case("tail_sync", "probe"),
    Case("h96_t8k_t16k", "probe"),
    Case("bf16_gate_params", "probe"),
    Case("profile_h96_t8k", "profile"),
    Case("profile_h96_t16k", "profile"),
)
CASE_BY_ID = {case.case_id: case for case in CASES}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-commit", default="unknown")
    parser.add_argument("--soc", default="ascend950")
    parser.add_argument("--device-visible-id", type=int, default=0)
    parser.add_argument("--cases", default="smoke")
    parser.add_argument("--case-timeout", type=int, default=900)
    parser.add_argument("--profile-launch-count", type=int, default=20)
    parser.add_argument("--profile-warm-up", type=int, default=5)
    return parser.parse_args()


def selected_cases(value: str) -> list[Case]:
    if value == "all":
        return list(CASES)
    if value == "smoke":
        return list(CASES[:3])
    case_ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(case_ids) - set(CASE_BY_ID))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    return [CASE_BY_ID[case_id] for case_id in case_ids]


def case_command(args, case: Case, case_dir: Path) -> list[str]:
    probe = args.repo_dir / "tests/operators/chunk_kda_fwd/st/probe_a5_tail.py"
    profile = args.repo_dir / "tests/operators/chunk_kda_fwd/performance/profile.py"
    if case.case_id == "tail_sync":
        return [sys.executable, "-u", str(probe), "--device", "0"]
    if case.case_id == "bf16_gate_params":
        return [
            sys.executable, "-u", str(probe), "--device", "0",
            "--bf16-gate-params",
        ]
    if case.case_id == "h96_t8k_t16k":
        return [
            sys.executable, "-u", str(probe), "--device", "0", "--long-seq",
        ]
    suffix = "t8k" if case.case_id.endswith("t8k") else "t16k"
    return [
        sys.executable,
        "-u",
        str(profile),
        "--case-id",
        f"chunk_kda_fwd_h96_{suffix}_model_performance",
        "--output",
        str(case_dir / "msopprof"),
        "--launch-count",
        str(args.profile_launch_count),
        "--warm-up",
        str(args.profile_warm_up),
        "--repeats",
        "2",
        "--kill",
        "off",
    ]


def classify_failure(log_text: str, returncode: int) -> tuple[str, str]:
    lowered = log_text.lower()
    if "out of memory" in lowered or "acl_error_rt_memory" in lowered:
        return "OOM", "device memory allocation failed"
    if returncode == 124 or any(
        line.lstrip().startswith("[TIMEOUT]") for line in log_text.splitlines()
    ):
        return "TIMEOUT", "case reported a timeout"
    if any(
        marker in lowered
        for marker in (
            "acl_error_rt_device_task_abort",
            "device task abort",
            "stream synchronize failed",
        )
    ):
        return "DEVICE_ERROR", "device task failed; stop before running more cases"
    if '"deterministic": false' in lowered:
        return "MISMATCH", "binary nondeterminism reported; see per-output details"
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    return "ERROR", lines[-1] if lines else f"exited with status {returncode}"


def run_logged(command, *, cwd: Path, env: dict, log_path: Path, timeout: int):
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return process.returncode, True


def extract_probe_records(log_text: str) -> list[dict]:
    records = []
    subcase = "unknown"
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[RUN] "):
            subcase = line[len("[RUN] "):]
            continue
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "output_count" not in payload or not isinstance(payload.get("outputs"), dict):
            continue
        attn_out = payload.get("outputs", {}).get("attn_out") or {}
        records.append({
            "subcase": subcase,
            "deterministic": payload.get("deterministic"),
            "deterministic_by_output": payload.get("deterministic_by_output"),
            "input_integrity": payload.get("input_integrity"),
            "attn_out_shape": attn_out.get("shape"),
            "attn_out_dtype": attn_out.get("dtype"),
            "runtime": payload.get("runtime"),
            "binary_differences": payload.get("binary_differences", [])[:16],
            "repeat_summaries": payload.get("repeat_summaries", []),
        })
    return records


def extract_probe_progress(log_text: str) -> dict | None:
    """Return the last flushed child stage, retaining its case context."""
    subcase = "unknown"
    context = {}
    last_progress = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[RUN] "):
            subcase = line[len("[RUN] "):]
            context = {}
            continue
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        stage = payload.get("stage")
        if not stage:
            continue
        if stage == "child_start":
            context = {
                key: payload.get(key)
                for key in (
                    "tokens", "heads", "layout", "adapter",
                    "final_state", "saved",
                )
            }
        last_progress = {
            "subcase": subcase,
            **context,
            "stage": stage,
        }
        if "repeat" in payload:
            last_progress["repeat"] = payload["repeat"]
        for key in ("gate_dtype", "a_log_dtype", "dt_bias_dtype"):
            if key in payload:
                last_progress[key] = payload[key]
    return last_progress


def format_probe_progress(progress: dict) -> str:
    ordered_keys = (
        "subcase", "stage", "repeat", "tokens", "heads", "layout",
        "adapter", "final_state", "saved", "gate_dtype", "a_log_dtype",
        "dt_bias_dtype",
    )
    return ",".join(
        f"{key}={progress[key]}"
        for key in ordered_keys
        if progress.get(key) is not None
    )


def compact_mismatch_note(records: list[dict], default: str) -> str:
    for record in records:
        differences = record.get("binary_differences") or []
        if not differences:
            continue
        first = differences[0]
        actual = first.get("actual") or {}
        baseline = first.get("baseline") or {}
        repeats = sorted({item.get("repeat") for item in differences})
        changed_outputs = sorted(
            name
            for name, is_equal in (record.get("deterministic_by_output") or {}).items()
            if is_equal is False
        )
        return (
            f"{record['subcase']}: {first.get('output')}"
            f"{first.get('first_index')} differs in repeats {repeats}; "
            f"bits {baseline.get('bits')} -> {actual.get('bits')}; "
            f"changed_outputs={changed_outputs}; "
            f"input_integrity={record.get('input_integrity')}"
        )
    return default


def write_summary(args, results):
    lines = [
        "KDA A5 acceptance summary",
        f"commit={args.repo_commit}",
        f"soc={args.soc}",
    ]
    for result in results:
        lines.append(
            f"case={result['case_id']} status={result['status']} "
            f"returncode={result['returncode']}"
        )
        records = result.get("probe_records") or []
        progress = result.get("probe_progress")
        if progress:
            lines.append(f"  last_progress={format_probe_progress(progress)}")
        for record in records:
            runtime = record.get("runtime") or {}
            lines.append(
                f"  subcase={record['subcase']} shape={record['attn_out_shape']} "
                f"dtype={record['attn_out_dtype']} "
                f"deterministic={record['deterministic']} "
                f"input_integrity={record['input_integrity']}"
            )
            if runtime:
                lines.append(
                    f"  runtime=torch:{runtime.get('torch')} "
                    f"torch_npu:{runtime.get('torch_npu')} "
                    f"triton_ascend:{runtime.get('triton_ascend')} "
                    f"device:{runtime.get('device_name')} "
                    f"cann:{runtime.get('ascend_home_path')}"
                )
            changed_outputs = sorted(
                name
                for name, is_equal in (
                    record.get("deterministic_by_output") or {}
                ).items()
                if is_equal is False
            )
            if changed_outputs:
                lines.append(f"  changed_outputs={changed_outputs}")
            differences = record.get("binary_differences") or []
            if differences:
                first_by_output = {}
                for difference in differences:
                    first_by_output.setdefault(
                        difference.get("output"), difference
                    )
                for output, first in first_by_output.items():
                    actual = first.get("actual") or {}
                    baseline = first.get("baseline") or {}
                    repeats = sorted({
                        item.get("repeat")
                        for item in differences
                        if item.get("output") == output
                    })
                    lines.append(
                        f"  first_diff={output}"
                        f"{first.get('first_index')} count="
                        f"{first.get('mismatched_elements')} repeats={repeats} "
                        f"baseline={baseline.get('value')}/{baseline.get('bits')} "
                        f"actual={actual.get('value')}/{actual.get('bits')}"
                    )
        if result["note"]:
            lines.append(f"  note={result['note']}")
    (args.output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_reports(args, results):
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_commit": args.repo_commit,
            "soc": args.soc,
            "device_visible_id": args.device_visible_id,
            "case_timeout": args.case_timeout,
        },
        "results": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# KDA A5 acceptance",
        "",
        f"- commit: `{args.repo_commit}`",
        f"- SOC: `{args.soc}`",
        "",
        "| case | kind | status | log | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['case_id']} | {result['kind']} | {result['status']} | "
            f"`{result['log']}` | {result['note']} |"
        )
    (args.output_dir / "results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_summary(args, results)


def main():
    args = parse_args()
    if args.case_timeout < 1:
        raise ValueError("--case-timeout must be positive")
    args.repo_dir = args.repo_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir()
    env = os.environ.copy()
    env["TEST_DEVICE_ID"] = str(args.device_visible_id)
    results = []
    failed = False
    for case in selected_cases(args.cases):
        case_dir = args.output_dir / case.case_id
        case_dir.mkdir()
        log_path = logs_dir / f"{case.case_id}.log"
        command = case_command(args, case, case_dir)
        print(f"[RUN] {case.case_id}", flush=True)
        returncode, timed_out = run_logged(
            command,
            cwd=args.repo_dir,
            env=env,
            log_path=log_path,
            timeout=args.case_timeout,
        )
        status = "PASS"
        note = ""
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if timed_out:
            status, note = "TIMEOUT", f"exceeded {args.case_timeout} seconds"
        elif returncode:
            status, note = classify_failure(
                log_text, returncode
            )
        probe_records = extract_probe_records(log_text)
        probe_progress = extract_probe_progress(log_text)
        if status == "MISMATCH":
            note = compact_mismatch_note(probe_records, note)
        elif status == "TIMEOUT" and probe_progress:
            note = f"{note}; last_progress={format_probe_progress(probe_progress)}"
        result = {
            **asdict(case),
            "status": status,
            "returncode": returncode,
            "command": shlex.join(command),
            "log": str(log_path),
            "note": note,
            "probe_records": probe_records,
            "probe_progress": probe_progress,
        }
        results.append(result)
        write_reports(args, results)
        print(f"[{status}] {case.case_id}", flush=True)
        if status != "PASS":
            failed = True
            if status in {"TIMEOUT", "OOM", "DEVICE_ERROR"}:
                print(f"Stop after fatal failure. See {log_path}", flush=True)
                return 1
            print(f"Continue after non-fatal failure. See {log_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
