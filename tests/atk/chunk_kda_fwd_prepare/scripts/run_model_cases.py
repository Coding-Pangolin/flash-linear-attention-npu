"""Run prepared model suites through native ATK on A2 or A5; retain raw reports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from prepare_model_cases import OP_DIR, load, sha256


def capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60, check=False)
    return dict(command=command, returncode=result.returncode, output=result.stdout)


def validate_plan(folder):
    plan = load(folder / "plan.json")
    if sha256(OP_DIR / "executor_chunk_kda_fwd_prepare.py") != plan["executor_sha256"]:
        raise ValueError("Executor changed since case preparation; regenerate the plan")
    if sha256(OP_DIR / "model_cases/manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Source model manifest changed; regenerate the plan")
    if sha256(OP_DIR / "atk_chunk_kda_fwd_prepare.json") != plan["templates_sha256"]:
        raise ValueError("Source input templates changed; regenerate the plan")
    for suite in plan["suites"]:
        for data in suite["files"].values():
            path = folder / data["file"]
            if path.name != data["file"] or sha256(path) != data["sha256"]:
                raise ValueError(f"Case file hash/path mismatch: {path}")
            if len(load(path)) != data["count"]:
                raise ValueError(f"Case count mismatch: {path}")
    return plan


def command_for(atk, stage, cases, output, device, timeout):
    command = [atk, "node", "--name", "npu_dut", "--backend", "npu",
               "--devices", str(device), "--output_path", str(output)]
    if stage == "accuracy":
        command += ["node", "--name", "cpu_golden", "--backend", "cpu",
                    "--output_path", str(output)]
    command += ["task", "-c", str(cases), "-p", str(OP_DIR / "executor_chunk_kda_fwd_prepare.py"),
                "--task", "accuracy" if stage == "accuracy" else "performance_device"]
    if stage == "accuracy":
        command += ["--bm_device", "cpu", "--gm_init_flag"]
    else:
        command += ["--save_data", "profile", "-sp"]
    return command + ["-to", str(timeout)]


def run_logged(command, folder, timeout):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "command.json").write_text(json.dumps(command, indent=2)+"\n", encoding="utf-8")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # End this launched task's descendants only; do not leave a timed-out
            # ATK DUT running while another performance group starts.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return dict(started_utc=started, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                returncode=code, command=command)


def accuracy_verdict(folder, expected):
    """Require positive, explicit, complete NPU summary evidence; never skip missing reports."""
    import openpyxl

    evidence = []
    for path in sorted(folder.rglob("*.xlsx")):
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if "summary" not in book.sheetnames:
                continue
            rows = list(book["summary"].values)
            if not rows:
                continue
            header = [str(v or "").strip() for v in rows[0]]
            required = ["总用例数", "执行成功用例个数", "执行失败用例个数", "通过用例个数", "精度是否达标"]
            if not all(name in header for name in required):
                continue
            indexes = [header.index(name) for name in required]
            for row in rows[1:]:
                if not any("npu_dut" in str(v) for v in row):
                    continue
                values = [row[i] if i < len(row) else None for i in indexes]
                total, executed, failed, passed, status = values
                try:
                    counts = tuple(int(v) for v in (total, executed, failed, passed))
                except (TypeError, ValueError):
                    continue
                good = counts == (expected, expected, 0, expected) and str(status).strip().lower() in {
                    "pass", "passed", "true", "通过", "达标"
                }
                evidence.append(dict(report=str(path), counts=counts, status=str(status), passed=good))
        finally:
            book.close()
    passed = bool(evidence) and all(e["passed"] for e in evidence)
    return dict(status="PASS" if passed else "NOT_VERIFIED", expected=expected, evidence=evidence,
                note="Missing/unknown report schema, partial execution or failed accuracy cannot pass")


def preflight(args):
    atk = shutil.which(args.atk)
    if not atk:
        raise RuntimeError("ATK executable not found")
    version = capture([atk, "--version"])
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version["output"])
    if version["returncode"] or not match or tuple(map(int, match.groups())) < (26, 8, 8):
        raise RuntimeError(f"ATK >= 26.8.8 required: {version['output']}")
    import openpyxl  # Check report dependency before long NPU/CPU work.
    del openpyxl
    board = capture(["npu-smi", "info", "-t", "board", "-i", str(args.device), "-c", "0"])
    detected = "A2" if re.search(r"910B", board["output"], re.I) else (
        "A5" if "950" in board["output"] else "unknown")
    if board["returncode"] or detected != args.soc:
        raise RuntimeError(f"Requested {args.soc}, detected {detected}: {board['output']}")
    package = capture([sys.executable, "-c",
                       "import torch,torch_npu,fla_npu; "
                       "print('torch',torch.__version__,'torch_npu',torch_npu.__version__); "
                       "print('fla_npu',fla_npu.__file__)"])
    if package["returncode"]:
        raise RuntimeError(f"Runtime import failed: {package['output']}")
    return atk, dict(soc=detected, device=args.device, build_label=args.build_label,
                     python=sys.executable, host=platform.platform(),
                     atk=version, board=board, packages=package,
                     wheel_sha256=sha256(args.wheel) if args.wheel else None,
                     host_load=os.getloadavg(), executor_sha256=sha256(OP_DIR / "executor_chunk_kda_fwd_prepare.py"),
                     environment={k:os.environ.get(k) for k in (
                         "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
                         "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH", "KDA_PREPARE_ATK_REFERENCE_BATCH_SIZE")})


def machine_state(device):
    return {kind:capture(["npu-smi", "info", "-t", kind, "-i", str(device), "-c", "0"])
            for kind in ("board", "sensors")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory; never reuse another run")
    parser.add_argument("--soc", required=True, choices=["A2", "A5"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-label", required=True, help="Installed wheel/source build identity")
    parser.add_argument("--wheel", type=Path, help="Installed wheel to fingerprint")
    parser.add_argument("--stage", choices=["all", "accuracy", "performance"], default="all")
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--case-timeout", type=int, default=14400)
    parser.add_argument("--job-timeout", type=int, default=28800)
    parser.add_argument("--atk", default="atk")
    parser.add_argument("--dry-run", action="store_true", help="Validate assets and save commands without hardware")
    args = parser.parse_args()
    if args.groups < 3 or args.device < 0 or args.case_timeout <= 0 or args.job_timeout <= 0:
        parser.error("Require groups>=3, device>=0 and positive timeouts")
    args.prepared, args.out = args.prepared.resolve(), args.out.resolve()
    plan = validate_plan(args.prepared)
    args.out.mkdir(parents=True, exist_ok=False)
    atk = args.atk
    result = dict(state="DRY_RUN" if args.dry_run else "RUNNING", soc=args.soc,
                  build_label=args.build_label, plan=plan, tasks=[],
                  scope="Prepare all13 outputs; full KDA forward/backward not tested",
                  acceptance="pending calibration and model configuration confirmation",
                  performance_method="native ATK performance_device, three independent groups; not historical Event p50")

    def save():
        (args.out / "run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    save()
    try:
        if not args.dry_run:
            if os.name != "posix":
                raise RuntimeError("Hardware execution requires the Linux Ascend environment")
            atk, result["machine"] = preflight(args)
        for stage in (["accuracy", "performance"] if args.stage == "all" else [args.stage]):
            for suite in plan["suites"]:
                data = suite["files"][stage]
                for group in range(1 if stage == "accuracy" else args.groups):
                    folder = args.out / stage / suite["profile"]["name"] / f"g{group+1}"
                    command = command_for(atk, stage, args.prepared / data["file"], folder,
                                          args.device, args.case_timeout)
                    task = dict(stage=stage, profile=suite["profile"]["name"], group=group+1,
                                expected_cases=data["count"], command=command, status="PLANNED")
                    result["tasks"].append(task)
                    save()
                    if args.dry_run:
                        continue
                    task["before"] = machine_state(args.device)
                    task.update(run_logged(command, folder, args.job_timeout))
                    task["after"] = machine_state(args.device)
                    if task["returncode"]:
                        raise RuntimeError(f"ATK returned {task['returncode']}; inspect {folder}")
                    if stage == "accuracy":
                        task["verdict"] = accuracy_verdict(folder, data["count"])
                        task["status"] = task["verdict"]["status"]
                        save()
                        if task["status"] != "PASS":
                            raise RuntimeError("Accuracy not verified; performance phase will not start")
                    else:
                        reports = [str(p) for p in folder.rglob("*.xlsx")]
                        task.update(status="MEASURED_PENDING_REPORT_REVIEW" if reports else "MISSING_REPORT",
                                    reports=reports)
                        if not reports:
                            raise RuntimeError("ATK returned zero without a performance report")
                    save()
        if not args.dry_run:
            result["state"] = "EXECUTED_PENDING_ACCEPTANCE"
    except BaseException as exc:
        result.update(state="STOPPED_NOT_ACCEPTED", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    save()
    print(json.dumps(dict(state=result["state"], tasks=len(result["tasks"]), output=str(args.out)), indent=2))


if __name__ == "__main__":
    main()
