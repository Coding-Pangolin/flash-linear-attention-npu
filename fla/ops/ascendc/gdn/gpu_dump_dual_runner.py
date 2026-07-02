"""Shared batch runner for GDN GPU dump dual benchmark scripts."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpu_dump_dual_utils import add_viz_cli_args, resolve_viz_dir


def add_skip_cli_args(parser: argparse.ArgumentParser) -> None:
    add_viz_cli_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run all cases even if report+vz artifacts already exist",
    )


def resolve_case_viz_dir(
    *,
    viz_dir: Path | None,
    pt_path: Path | None,
    case_dir: Path | None,
    case_label: str,
    default_report_dir: Path,
) -> Path:
    base = resolve_viz_dir(
        viz_dir=viz_dir,
        pt_path=pt_path,
        case_dir=case_dir,
        default_report_dir=default_report_dir,
    )
    return (base or default_report_dir / "viz") / case_label.replace("/", "_")


def load_report_index(report_path: Path | None) -> dict[str, dict[str, Any]]:
    if report_path is None or not report_path.is_file():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for item in data.get("results", []):
        case_key = item.get("case")
        if case_key:
            index[str(case_key)] = item
    return index


def viz_artifacts_ready(
    viz_case_dir: Path,
    tensor_names: tuple[str, ...],
    *,
    require_viz: bool,
) -> bool:
    if not require_viz:
        return True
    if not viz_case_dir.is_dir():
        return False
    for name in tensor_names:
        if not any(viz_case_dir.glob(f"{name}*_Standard.png")):
            return False
    return True


def should_skip_case(
    case_key: str,
    viz_case_dir: Path,
    existing_by_case: dict[str, dict[str, Any]],
    tensor_names: tuple[str, ...],
    *,
    skip_enabled: bool,
    require_viz: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if not skip_enabled:
        return False, None
    prev = existing_by_case.get(case_key)
    if prev is None or prev.get("status") != "pass":
        return False, None
    if not viz_artifacts_ready(viz_case_dir, tensor_names, require_viz=require_viz):
        return False, None
    skipped = dict(prev)
    skipped["status"] = "skipped"
    skipped["skip_reason"] = "existing pass report and viz artifacts"
    return True, skipped


def summarize_results(results: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for r in results if r.get("status") == "pass")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "fail")
    return {
        "total": len(results),
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "ok": passed + skipped,
    }


def write_report(
    report_path: Path,
    *,
    op_name: str,
    mode: str,
    results: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = summarize_results(results)
    report: dict[str, Any] = {
        "op": op_name,
        "mode": mode,
        "total": stats["total"],
        "passed": stats["passed"],
        "skipped": stats["skipped"],
        "failed": stats["failed"],
        "results": results,
    }
    if extra:
        report.update(extra)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_dual_batch(
    args: argparse.Namespace,
    *,
    op_name: str,
    report_basename: str,
    viz_tensor_names: tuple[str, ...],
    collect_pt_paths: Callable[[argparse.Namespace], list[Path]],
    select_cases: Callable[[Path, argparse.Namespace], list[Path]],
    run_one_pt: Callable[..., dict[str, Any]],
    run_one_case: Callable[..., dict[str, Any]],
    run_case_kwargs: dict[str, Any] | None = None,
    report_extra: dict[str, Any] | None = None,
) -> int:
    pt_paths = collect_pt_paths(args)
    enable_viz = not args.no_viz
    sample_count = args.sample_count
    skip_enabled = not args.force
    run_case_kwargs = run_case_kwargs or {}

    results: list[dict[str, Any]] = []
    failed = 0
    skipped_count = 0

    if pt_paths:
        default_report_dir = pt_paths[0].resolve().parent
        report_path = args.report or (default_report_dir / report_basename)
        existing_by_case = load_report_index(report_path)

        for pt_path in pt_paths:
            label = pt_path.name
            viz_case_dir = resolve_case_viz_dir(
                viz_dir=args.viz_dir,
                pt_path=pt_path,
                case_dir=None,
                case_label=label,
                default_report_dir=pt_path.parent,
            )
            do_skip, skipped_entry = should_skip_case(
                label,
                viz_case_dir,
                existing_by_case,
                viz_tensor_names,
                skip_enabled=skip_enabled,
                require_viz=enable_viz,
            )
            if do_skip and skipped_entry is not None:
                skipped_count += 1
                print(f"\n=== {label} SKIPPED (already passed) ===", flush=True)
                print(f"  report: {report_path}", flush=True)
                print(f"  viz:    {viz_case_dir}", flush=True)
                results.append(skipped_entry)
                continue

            try:
                results.append(run_one_pt(
                    pt_path,
                    label=label,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {label} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({
                    "case": label,
                    "status": "fail",
                    "pt": str(pt_path),
                    "error": str(e),
                })

        mode = "pt"
        extra = dict(report_extra or {})
    else:
        if args.dump_root is None:
            print("ERROR: provide --dump-root or --pt/--pts", file=sys.stderr)
            return 2
        selected = select_cases(args.dump_root, args)
        if not selected:
            print("No cases selected.", file=sys.stderr)
            return 1

        default_report_dir = args.dump_root
        report_path = args.report or (default_report_dir / report_basename)
        existing_by_case = load_report_index(report_path)

        for case_dir in selected:
            label = case_dir.name
            viz_case_dir = resolve_case_viz_dir(
                viz_dir=args.viz_dir,
                pt_path=None,
                case_dir=case_dir,
                case_label=label,
                default_report_dir=case_dir,
            )
            do_skip, skipped_entry = should_skip_case(
                label,
                viz_case_dir,
                existing_by_case,
                viz_tensor_names,
                skip_enabled=skip_enabled,
                require_viz=enable_viz,
            )
            if do_skip and skipped_entry is not None:
                skipped_count += 1
                print(f"\n=== {label} SKIPPED (already passed) ===", flush=True)
                print(f"  report: {report_path}", flush=True)
                print(f"  viz:    {viz_case_dir}", flush=True)
                results.append(skipped_entry)
                continue

            try:
                results.append(run_one_case(
                    case_dir,
                    verbose=True,
                    enable_viz=enable_viz,
                    sample_count=sample_count,
                    viz_dir=args.viz_dir,
                    **run_case_kwargs,
                ))
            except Exception as e:
                failed += 1
                print(f"\n=== {case_dir.name} FAILED ===\n{e}", flush=True)
                traceback.print_exc()
                results.append({
                    "case": case_dir.name,
                    "status": "fail",
                    "error": str(e),
                })

        mode = "case_dir"
        extra = dict(report_extra or {})
        extra.setdefault("dump_root", str(args.dump_root))

    report = write_report(
        report_path,
        op_name=op_name,
        mode=mode,
        results=results,
        extra=extra,
    )
    print(
        f"\nDone: {report['passed']} passed, {report['skipped']} skipped, "
        f"{report['failed']} failed / {report['total']} total",
        flush=True,
    )
    print(f"report -> {report_path}", flush=True)
    if skipped_count:
        print(f"(use --force to re-run {skipped_count} skipped case(s))", flush=True)
    return 1 if failed else 0
