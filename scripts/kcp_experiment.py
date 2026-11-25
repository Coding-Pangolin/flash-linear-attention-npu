#!/usr/bin/env python3
"""CPU-only KCP resource planning, schedule checks and profiler aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

# Load the packaged CPU-only accounting without importing fla_npu/torch.
import importlib.util
_RESOURCE_PATH = (Path(__file__).resolve().parents[1] /
    "torch_custom/fla_npu/fla_npu/ops/ascendc/_kcp_resources.py")
_RESOURCE_SPEC = importlib.util.spec_from_file_location("kcp_resources", _RESOURCE_PATH)
_RESOURCE_MODULE = importlib.util.module_from_spec(_RESOURCE_SPEC)
_RESOURCE_SPEC.loader.exec_module(_RESOURCE_MODULE)
LIMIT = _RESOURCE_MODULE.LIMIT
memory_plan = _RESOURCE_MODULE.memory_plan


def schedule(heads, group):
    if min(heads, group) <= 0:
        raise ValueError("heads and group must be positive")
    count = (heads + group - 1) // group
    operations = []
    for i in range(count):
        operations.extend([("produce", i), ("submit", i)])
        if i:
            operations.append(("consume", i - 1))
    operations.append(("consume", count - 1))
    live, posted, consumed = set(), set(), set()
    for action, i in operations:
        if action == "produce":
            live.add(i)
            if len(live) > 2:
                raise AssertionError("more than two producer groups live")
        elif action == "submit":
            if i not in live:
                raise AssertionError("communication before production")
            posted.add(i)
        else:
            if i not in posted or i in consumed:
                raise AssertionError("invalid consume")
            consumed.add(i)
            live.remove(i)
    if live or len(consumed) != count:
        raise AssertionError("schedule did not drain")
    return operations


def aggregate_windows(records, *, expected_ranks=None, expected_iterations=None, expected_windows=None):
    """Join by window/rank/iteration, retaining slow ranks and complete windows."""
    names = {row.get("backend") for row in records}
    if len(names) > 1:
        if None in names:
            raise ValueError("backend is missing from some records")
        return {"backends": {name: aggregate_windows(
            [row for row in records if row["backend"] == name],
            expected_ranks=expected_ranks, expected_iterations=expected_iterations,
            expected_windows=expected_windows) for name in sorted(names)}}
    windows = {}
    for row in records:
        key = (str(row["window"]), int(row["rank"]), int(row["iteration"]))
        duration = float(row["duration_us"])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if key in windows:
            raise ValueError("duplicate window/rank/iteration")
        windows[key] = duration
    if not windows:
        raise ValueError("no measurements")
    rank_ids = {k[1] for k in windows}
    if rank_ids != set(range(len(rank_ids))):
        raise ValueError("rank IDs must be complete and zero based")
    if expected_ranks is not None and rank_ids != set(range(expected_ranks)):
        raise ValueError("unexpected rank set")
    if expected_windows is not None and {k[0] for k in windows} != {str(i) for i in range(expected_windows)}:
        raise ValueError("unexpected window set")
    summaries = []
    expected_iters = None
    for window in sorted({k[0] for k in windows}):
        means = {}
        for rank in sorted(rank_ids):
            current = {k[2]: v for k, v in windows.items() if k[:2] == (window, rank)}
            if not current:
                raise ValueError("missing rank in window")
            if expected_iters is None:
                expected_iters = set(current)
                required = (set(range(expected_iterations)) if expected_iterations is not None
                            else set(range(max(expected_iters) + 1)))
                if expected_iters != required:
                    raise ValueError("iterations must be complete and zero based")
            if set(current) != expected_iters:
                raise ValueError("incomplete rank/window iteration set")
            means[rank] = statistics.mean(current.values())
        summaries.append({"window": window, "rank_mean_us": means,
                          "max_rank_mean_us": max(means.values())})
    values = [w["max_rank_mean_us"] for w in summaries]
    return {"windows": summaries, "median_max_rank_mean_us": statistics.median(values),
            "relative_window_spread": (max(values) - min(values)) / max(statistics.median(values), 1e-30)}


def profiler_tasks(path, kernel):
    """Exact-name task extraction; deliberately not an end-to-end makespan."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Op Name") != kernel:
                continue
            field = next((k for k in row if k.replace(" ", "").lower() == "taskduration(us)"), None)
            if field is None:
                raise ValueError("Task Duration(us) column missing")
            duration = float(row[field])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid profiler duration")
            rows.append(duration)
    if not rows:
        raise ValueError("no exact kernel matches")
    return {"kernel": kernel, "count": len(rows), "mean_us": statistics.mean(rows),
            "last5_mean_us": statistics.mean(rows[-5:]), "is_joined_makespan": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("memory")
    for name in ("tokens", "heads", "cp", "group"):
        m.add_argument("--" + name, type=int, required=True)
    m.add_argument("--training", action="store_true")
    m.add_argument("--runtime-bytes", type=int, default=0)
    m.add_argument("--communication-bytes", type=int, default=0)
    m.add_argument("--output-final-state", action="store_true")
    m.add_argument("--no-bias", dest="has_bias", action="store_false")
    m.add_argument("--external-beta-sigmoid", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("input", type=Path)
    a = sub.add_parser("tasks")
    a.add_argument("input", type=Path)
    a.add_argument("--kernel", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "memory":
        result = memory_plan(**args)
    elif command == "aggregate":
        result = aggregate_windows(json.loads(args["input"].read_text(encoding="utf-8")))
    else:
        result = profiler_tasks(args["input"], args["kernel"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
