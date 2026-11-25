#!/usr/bin/env python3
"""Plan the 48 core cases, or run one SoC/CP slice on explicitly chosen devices.

Source CANN and activate the private test venv first. Without --run this only
prints JSON and never imports torch or touches a device. Memory values are
estimates; allocator peaks do not include all HCCL/driver reservations.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from kcp_experiment import memory_plan

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((512, 8), (2048, 16), (8192, 32), (16384, 64))


def cases(runtime_bytes=0, communication_bytes=0, group_size=4):
    result = []
    for soc in ("a2", "a5"):
        for cp in (2, 4, 8):
            for tokens, heads in SHAPES:
                for mode in ("none", "save"):
                    candidates = [g for g in (1, 2, 4, 8, 16) if g <= group_size and g <= heads]
                    legal = [g for g in candidates if memory_plan(tokens, heads, cp, g,
                        training=mode == "save", runtime_bytes=runtime_bytes,
                        communication_bytes=communication_bytes, output_final_state=True,
                        external_beta_sigmoid=soc == "a5" and mode == "save")["within_temporary_budget"]]
                    selected = max(legal) if legal else None
                    result.append(dict(soc=soc, cp=cp, tokens=tokens, heads=heads,
                        mode=mode, group_size=selected, resources=None if selected is None else
                        memory_plan(tokens, heads, cp, selected, training=mode == "save",
                            runtime_bytes=runtime_bytes, communication_bytes=communication_bytes,
                            output_final_state=True, external_beta_sigmoid=soc == "a5" and mode == "save")))
    return result


def available_devices(devices):
    """Fail closed on ambiguous health/process inventory; do not reset devices."""
    inventory = {}
    for device in devices:
        outputs = {}
        for kind in ("health", "proc-mem"):
            p = subprocess.run(["npu-smi", "info", "-t", kind, "-i", str(device)],
                               capture_output=True, text=True, timeout=20, check=True)
            outputs[kind] = p.stdout
        health = re.findall(r"Health(?: Status)?\s*:\s*(\w+)", outputs["health"])
        if not health or any(value != "OK" for value in health):
            raise RuntimeError(f"device {device} is unhealthy or health is ambiguous: {outputs['health']}")
        if not any(message in outputs["proc-mem"].lower()
                   for message in ("no process in device", "no running processes found")):
            raise RuntimeError(f"device {device} has processes or inventory is ambiguous: {outputs['proc-mem']}")
        inventory[str(device)] = outputs
    return inventory


def execute(command, env, directory, name, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + "_command.json")).write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (directory / (name + ".log")).open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Kill only this run's newly created process group, including
            # torchrun workers; killing just the parent can orphan kernels.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--soc", choices=("a2", "a5"))
    p.add_argument("--devices", help="physical device IDs, for example 0,7")
    p.add_argument("--group-size", type=int, choices=(1, 2, 4, 8, 16), default=4)
    p.add_argument("--backends", default="staged,pipeline",
                   help="comma-separated staged,pipeline,mc2; mc2 first runs its native gate")
    p.add_argument("--runtime-bytes", type=int, default=0)
    p.add_argument("--communication-bytes", type=int, default=0)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/kcp_matrix")
    a = p.parse_args()
    backends = a.backends.split(",")
    if (not backends or len(set(backends)) != len(backends)
            or any(value not in ("staged", "pipeline", "mc2") for value in backends)):
        p.error("--backends must contain distinct staged,pipeline,mc2 values")
    planned = cases(a.runtime_bytes, a.communication_bytes, a.group_size)
    if not a.run:
        print(json.dumps({"count": len(planned), "cases": planned}, indent=2))
        return
    if not a.soc or not a.devices:
        p.error("--run requires --soc and --devices")
    if a.soc == "a5" and "mc2" in backends:
        p.error("A5 native MC2 is gated until communication resources pass; use check_kcp_native.py for diagnosis")
    if os.name != "posix":
        p.error("device execution requires Linux; planning works on every host")
    devices = [int(x) for x in a.devices.split(",")]
    if len(devices) not in (2, 4, 8) or len(devices) != len(set(devices)) or min(devices) < 0:
        p.error("choose 2, 4 or 8 distinct nonnegative devices")
    # One task-specific directory prevents accidental overwrite of prior runs.
    target = a.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{a.soc}_cp{len(devices)}")
    target.mkdir(parents=True, exist_ok=False)
    inventory = available_devices(devices)
    (target / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, devices)),
               HCCL_EXEC_TIMEOUT="60", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(devices)}", "examples/cp/benchmark_kda_cp.py"]
    if "mc2" in backends:
        native = command[:-1] + ["examples/cp/check_kcp_native.py", "--heads", "16",
            "--negative", "--resources", "--output", str(target / "native_gate")]
        execute(native, env, target, "native_gate", 60)
    for mode in ("none", "save"):
        check = command + ["--tokens", "65", "--heads", "3", "--group-size", "2",
            "--mode", mode, "--accuracy", "--accuracy-only", "--stress-repeats", "8",
            "--negative-preflight", "--backends", a.backends,
            "--output", str(target / ("accuracy_" + mode))]
        if mode == "save":
            check.append("--backward")
        execute(check, env, target, "accuracy_" + mode, 60)
    selected = [case for case in planned if case["soc"] == a.soc and case["cp"] == len(devices)]
    (target / "plan.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    for case in selected:
        if case["group_size"] is None:
            raise RuntimeError(f"no group satisfies the estimated scratch budget: {case}")
        available_devices(devices)
        name = f"t{case['tokens']}_h{case['heads']}_{case['mode']}_g{case['group_size']}"
        perf = command + ["--tokens", str(case["tokens"]), "--heads", str(case["heads"]),
            "--mode", case["mode"], "--group-size", str(case["group_size"]),
            "--backends", a.backends,
            "--warmup", "20", "--iterations", "100", "--windows", "5",
            "--output", str(target / name)]
        print(json.dumps({"phase": "run_case", "case": name}), flush=True)
        execute(perf, env, target, name, 1800)
    print(target)


if __name__ == "__main__":
    main()
