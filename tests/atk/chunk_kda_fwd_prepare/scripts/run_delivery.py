"""Run fresh ATK acceptance with explicit identities. No local tensor caches.

The shared FLA_ATK/run_test_cpu.sh also accepts this directory. This entry adds
optional seed expansion, synchronized native dual roles, fresh directories and evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from delivery_cases import ROOT, OP, attr, expand_seeds, load, validate, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_python_identity():
    # Record the actual private Python adapter as well as the OPP/API hashes.
    # A .run package does not contain these independently installed wrappers.
    spec = importlib.util.find_spec("fla_npu")
    if spec is None or spec.origin is None:
        return {}
    root = Path(spec.origin).resolve().parent
    names = ("__init__.py", "libfla_npu_stable.so", "ops/ascendc/_stable.py",
             "ops/ascendc/_aclnn_ctypes.py", "ops/ascendc/_runtime.py")
    return dict(root=str(root), sha256={name:digest(root/name)
                for name in names if (root/name).is_file()})


def check_accuracy(reports, cases, console=None):
    """Fail closed on missing, duplicate, failed or wrong-output-count reports."""
    records = [row for path in reports for row in load(path)]
    if len(records) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} report rows, got {len(records)}")
    by_id = {int(row["编号"]): row for row in records}
    if len(by_id) != len(records):
        raise RuntimeError("Duplicate report IDs")
    if console is not None:
        with Path(console).open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if re.search(r"KDA_PREPARE_DUT_ERROR case=\d+", line):
                    raise RuntimeError(f"DUT exception recorded: {line.strip()}")
    passed = 0
    outputs = 0
    masks = {"none": 6, "recompute": 12, "save": 13, "forward": 7}
    for case in cases:
        row = by_id.get(case["id"])
        if row is None:
            raise RuntimeError(f"Missing case {case['id']}")
        if row.get("运行结果") != "SUCCESS":
            raise RuntimeError(f"Native execution failed case {case['id']}: {row.get('运行结果')}")
        detail = row.get("精度详情", {})
        if isinstance(detail, str):
            detail = ast.literal_eval(detail)
        compared = detail.get("cpu_cpu_reference", [])
        count = masks[attr(case, "backward_mode")]
        expected_names = {f"output_{i}.pt" for i in range(count)}
        actual_names = [item.get("filename") for item in compared]
        if (len(compared) != count or set(actual_names) != expected_names
                or not all(x.get("result") is True for x in compared)):
            raise RuntimeError(f"Native precision failure case {case['id']}: {compared}")
        # Numerical rows alone cannot excuse execution errors.
        reason = row.get("失败原因")
        if reason and str(reason).strip().lower() not in {"none", "nan", "-", "无"}:
            raise RuntimeError(f"Native execution error: {reason}")
        passed += 1
        outputs += count
    return dict(passed=passed, outputs_compared=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("accuracy", "model-accuracy", "performance",
        "determinism", "mssanitizer", "route-trace"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW directory; refuses reuse")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--port", type=int, default=19091, help="Task-private ATK worker port")
    parser.add_argument("--keep-outputs", action="store_true", help="Save ATK output:bin for debugging")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--package", type=Path, required=True, help="Exact .run/wheel artifact")
    parser.add_argument("--kernel-lib", type=Path, required=True, help="Loaded op API shared library")
    parser.add_argument("--tool", choices=("memcheck", "racecheck", "initcheck", "synccheck"), default="memcheck")
    parser.add_argument("--limit", type=int, help="Smoke only; never full acceptance")
    parser.add_argument("--case-file", type=Path, help="Explicit public-input subset; always smoke-only")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Explicit precision seed expansion; accuracy keeps the original 200 cases by default")
    parser.add_argument("--include-paths", action="store_true",
                        help="Opt in to 10 extra shapes; default DC/MSS is one case per template key")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--precision-jobs", type=int, choices=(1, 2, 3), default=1,
                        help="Native ATK case concurrency for accuracy only; no reduction of cases/outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.precision_jobs != 1 and args.scope not in {"accuracy", "model-accuracy"}:
        parser.error("--precision-jobs only applies to accuracy scopes")
    if args.seeds is not None and (args.scope not in {"accuracy", "model-accuracy"}
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("--seeds requires a precision scope and distinct seeds")
    if args.include_paths and (args.case_file is not None or args.scope not in {
            "determinism", "mssanitizer", "route-trace"}):
        parser.error("--include-paths requires a default DC/MSS/route-trace matrix")
    validate()
    suffix = {"accuracy": "", "model-accuracy": "_perf", "performance": "_perf",
              "determinism": "_mss", "mssanitizer": "_mss", "route-trace": "_mss"}[args.scope]
    cases = load(args.case_file or ROOT / f"atk_{OP}{suffix}.json")
    if args.seeds is not None:
        cases = expand_seeds(cases, tuple(args.seeds))
    elif args.scope in {"determinism", "mssanitizer", "route-trace"}:
        if args.include_paths:
            cases += load(ROOT / f"atk_{OP}_paths.json")
    if args.limit:
        cases = cases[:args.limit]
    out = args.out.resolve()
    # No overwrite, no stale input/output reuse, even for a dry run.
    out.mkdir(parents=True, exist_ok=False)
    case_path = out / f"atk_{OP}.json"
    write_json(case_path, cases)
    results = out / "results"
    cmd = [sys.executable, "-B", "-m", "atk", "node", "--name", "npu_dut",
           "--backend", "npu", "--port", str(args.port), "--devices", str(args.device), "--output_path", str(results)]
    accuracy = args.scope in {"accuracy", "model-accuracy"}
    if accuracy:
        cmd += ["node", "--name", "cpu_reference", "--backend", "cpu",
                "--port", str(args.port), "--output_path", str(results)]
    task = "accuracy" if accuracy else {"performance": "performance_device",
                "determinism": "accuracy_dc", "mssanitizer": "run", "route-trace": "run"}[args.scope]
    cmd += ["task", "-c", str(case_path), "-p", str(ROOT / f"executor_{OP}.py"),
            "--task", task, "-to", str(args.timeout), "-mt", str(args.precision_jobs),
            "--save_report_json", "--disable_id_seed", "--bind_cpu_type", "0"]
    if args.keep_outputs:
        cmd += ["--save_data", "output:bin", "-sp"]
    if accuracy:
        cmd += ["--bm_device", "cpu", "--syc_dataset", "--gm_init_flag"]
    elif args.scope == "determinism":
        cmd += ["--dc_loop_nums", "50"]
    elif args.scope == "performance":
        cmd += ["--save_data", "profile", "-sp", "--performance_data", "5,50,50"]
    elif args.scope == "mssanitizer":
        sanitizer_log = out / "sanitizer.log"
        sanitizer_log.touch(exist_ok=False)
        cmd += ["--mssanitizer", "-msl", str(sanitizer_log)]
        # ATK must read the actual tool log, not an unrelated empty placeholder.
        cmd = ["mssanitizer", f"--tool={args.tool}",
               "--log-level", "info", "--log-file", str(sanitizer_log), "--"] + cmd
    evidence = dict(started=datetime.now(timezone.utc).isoformat(), scope=args.scope,
        source_commit=args.source_commit, package_sha256=digest(args.package),
        api_library_sha256=digest(args.kernel_lib), device=args.device,
        runtime_python=runtime_python_identity(),
        cases=len(cases), smoke_only=args.limit is not None or args.case_file is not None, command=cmd,
        include_paths=args.include_paths,
        seed_override=args.seeds,
        precision_jobs=args.precision_jobs,
        effective_seeds=sorted({case["default_seed"] for case in cases}),
        executor_sha256=digest(ROOT / f"executor_{OP}.py"),
        runner_sha256=digest(Path(__file__)),
        environment={name: os.environ.get(name) for name in (
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_CACHE_PATH", "ASCEND_GLOBAL_LOG_LEVEL",
            "ASCEND_SLOG_PRINT_TO_STDOUT", "ASCEND_MODULE_LOG_LEVEL", "FLA_NPU_OP_API_LIB", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        case_sha256=digest(case_path), acceptance="not_run",
        requires=["actual tiling key trace", "device/compiler/runtime identity",
                  "matching instrumented kernel for each sanitizer tool"])
    write_json(out / "run.json", evidence)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if accuracy:
        module = importlib.import_module("atk.tasks.post_process.cv_fused_double_benchmark_compare")
        evidence["native_comparator_sha256"] = digest(module.__file__)
    for name, command in (("device.txt", ["npu-smi", "info"]),
                          ("atk.txt", [sys.executable, "-m", "atk", "--version"])):
        with (out / name).open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write_json(out / "run.json", evidence)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (out / "console.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(cmd, cwd=out, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT)
    evidence["exit_code"] = completed.returncode
    write_json(out / "run.json", evidence)
    if completed.returncode:
        evidence["acceptance"] = "failed"
        write_json(out / "run.json", evidence)
        raise RuntimeError(f"ATK failed: {out / 'console.log'}")
    if accuracy:
        try:
            evidence.update(check_accuracy(list(results.glob("atk_output/*/report/*.json")), cases,
                                           out / "console.log"))
        except Exception as exc:
            evidence.update(acceptance="failed", error=str(exc))
            write_json(out / "run.json", evidence)
            raise
        evidence["acceptance"] = "smoke_pass" if evidence["smoke_only"] else "native_accuracy_pass"
    else:
        # ATK exit 0 is NOT a DC/MSS/performance acceptance result. Preserve reports
        # for the shared checker and actual-key/instrumentation audit.
        evidence["acceptance"] = "reports_pending_review"
    write_json(out / "run.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
