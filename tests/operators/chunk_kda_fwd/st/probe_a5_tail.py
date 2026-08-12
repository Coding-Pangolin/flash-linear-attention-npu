#!/usr/bin/env python3
"""Isolate A5 KDA tail/final-state hangs with short, reproducible cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SHORT_CASES = (
    ("t64_final", 64, True, False),
    ("t65_minimal", 65, False, False),
    ("t65_final", 65, True, False),
    ("t65_saved", 65, True, True),
)
LONG_CASES = (
    ("h96_t8k", 8192, True, False),
    ("h96_t16k", 16384, True, False),
)
ADAPTER_CASES = (("bf16_gate_params", 64, True, False),)
OUTPUT_NAMES = (
    "attn_out", "final_state", "gk", "Aqk", "Akk", "w",
    "u", "qg", "kg", "v_new", "h", "initial_state",
)


def _stage(name, **details):
    print(json.dumps({"stage": name, **details}), flush=True)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--long-seq", action="store_true")
    parser.add_argument("--bf16-gate-params", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=65, help=argparse.SUPPRESS)
    parser.add_argument("--final-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--saved", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--constant-inputs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=("NTD", "BSND"), default="NTD", help=argparse.SUPPRESS)
    parser.add_argument("--adapter", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gate-params-bf16", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=2, help=argparse.SUPPRESS)
    return parser.parse_args()


def _fingerprint(torch, tensor):
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    stride = max(1, (flat.numel() + 4095) // 4096)
    value = flat[::stride][:4096].float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", "", 1),
        "finite": bool(torch.isfinite(flat).all().item()),
        "sum": float(value.sum().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
        "sample_numel": value.numel(),
    }


def _cpu_snapshot(outputs):
    return tuple(
        None if value is None else value.detach().cpu().contiguous()
        for value in outputs
    )


def _scalar_record(torch, tensor, index):
    scalar = tensor[index].reshape(1)
    integer_dtype = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }[scalar.element_size()]
    raw = int(scalar.view(integer_dtype).item())
    raw &= (1 << (scalar.element_size() * 8)) - 1
    return {
        "value": float(scalar.float().item()),
        "bits": f"0x{raw:0{scalar.element_size() * 2}x}",
    }


def _last_dim_neighborhood(torch, tensor, index, radius=3):
    if not index:
        return []
    center = index[-1]
    begin = max(0, center - radius)
    end = min(tensor.shape[-1], center + radius + 1)
    result = []
    for position in range(begin, end):
        item_index = (*index[:-1], position)
        result.append({
            "index": list(item_index),
            **_scalar_record(torch, tensor, item_index),
        })
    return result


def _tail_sample(torch, tensor):
    if tensor is None or tensor.numel() == 0:
        return None
    rows = tensor.detach().reshape(-1, tensor.shape[-1]).cpu().contiguous()
    channels = sorted({0, 1, 63, 64, 86, rows.shape[-1] - 1})
    return {
        str(channel): _scalar_record(torch, rows, (-1, channel))
        for channel in channels
        if 0 <= channel < rows.shape[-1]
    }


def _compare_snapshots(torch, current, baseline, names, repeat):
    equal_by_output = {}
    differences = []
    for name, value, expected in zip(names, current, baseline):
        if value is None or expected is None:
            is_equal = value is expected
        else:
            is_equal = value.shape == expected.shape and value.equal(expected)
        equal_by_output[name] = is_equal
        if is_equal:
            continue
        detail = {"output": name, "repeat": repeat}
        if value is None or expected is None:
            detail["optional_output_mismatch"] = True
        elif value.shape != expected.shape:
            detail["shape"] = list(value.shape)
            detail["baseline_shape"] = list(expected.shape)
        else:
            unequal = value != expected
            first = unequal.nonzero(as_tuple=False)[0]
            index = tuple(int(item) for item in first.tolist())
            detail.update({
                "mismatched_elements": int(unequal.sum().item()),
                "max_abs": float(
                    (value.float() - expected.float()).abs().max().item()
                ),
                "first_index": list(index),
                "actual": _scalar_record(torch, value, index),
                "baseline": _scalar_record(torch, expected, index),
                "actual_neighborhood": _last_dim_neighborhood(
                    torch, value, index
                ),
                "baseline_neighborhood": _last_dim_neighborhood(
                    torch, expected, index
                ),
            })
        differences.append(detail)
    return all(equal_by_output.values()), equal_by_output, differences


def _run_child(args):
    _stage(
        "child_start", tokens=args.tokens, heads=args.heads,
        layout=args.layout, adapter=args.adapter,
        final_state=args.final_state, saved=args.saved,
    )
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata

    import torch
    import torch_npu
    import fla_npu

    from fla_npu.ops.ascendc import chunk_kda_fwd

    _stage("imports_ready")

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.manual_seed(20260806 + args.heads + args.tokens)
    h, t, dim = args.heads, args.tokens, 128
    shape = (1, t, h, dim) if args.layout == "BSND" else (h, t, dim)
    beta_shape = (1, t, h) if args.layout == "BSND" else (h, t)
    if args.constant_inputs:
        q = torch.full(shape, dim**-0.5, dtype=torch.bfloat16, device=device)
        k = torch.full_like(q, dim**-0.5)
        v = torch.zeros_like(q)
        beta = torch.full(beta_shape, 0.5, dtype=torch.bfloat16, device=device)
        g = torch.full(shape, -1.0, dtype=torch.float32, device=device)
        a_log = torch.zeros(h, dtype=torch.float32, device=device)
        dt_bias = torch.zeros(h * dim, dtype=torch.float32, device=device)
    else:
        q = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        k = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        v = (torch.randn(shape) * 0.02).to(torch.bfloat16).to(device)
        beta = torch.sigmoid(torch.randn(beta_shape)).to(torch.bfloat16).to(device)
        g = (torch.randn(shape) * 0.2).to(device)
        a_log = (torch.randn(h) * 0.2 - 0.5).to(device)
        dt_bias = (torch.randn(h * dim) * 0.1).to(device)
    if args.gate_params_bf16:
        a_log = a_log.to(torch.bfloat16)
        dt_bias = dt_bias.to(torch.bfloat16)
    _stage(
        "inputs_ready", gate_dtype=str(g.dtype),
        a_log_dtype=str(a_log.dtype), dt_bias_dtype=str(dt_bias.dtype),
    )

    input_names = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
    input_values = (q, k, v, beta, g, a_log, dt_bias)
    input_baseline = _cpu_snapshot(input_values) if args.repeats > 1 else None
    baseline = None
    deterministic = None
    deterministic_by_output = None
    binary_differences = []
    fingerprints = None
    repeat_summaries = []
    started = time.perf_counter()
    for repeat in range(1, args.repeats + 1):
        _stage("launch_begin", repeat=repeat)
        common = {
            "cu_seqlens": None if args.layout == "BSND" else (0, t),
            "output_final_state": args.final_state,
            "safe_gate": True,
            "lower_bound": -5.0,
            "use_gate_in_kernel": True,
            "A_log": a_log,
            "dt_bias": dt_bias,
            "disable_recompute": args.saved,
            "return_intermediate_states": args.saved,
        }
        if args.adapter:
            from fla_npu.adapters.triton_ascend_kda import (
                triton_ascend_chunk_kda_fwd,
            )

            adapter_common = dict(common)
            adapter_common.pop("output_final_state")
            outputs = triton_ascend_chunk_kda_fwd(
                q, k, v, g, beta, dim**-0.5, None, args.final_state,
                chunk_size=64,
                transpose_state_layout=False,
                **adapter_common,
            )
        else:
            outputs = chunk_kda_fwd(
                q, k, v, g, beta, dim**-0.5, 64,
                layout=args.layout,
                **common,
            )
        _stage("launch_returned", repeat=repeat, output_count=len(outputs))
        torch.npu.synchronize()
        _stage("synchronize_done", repeat=repeat)
        fingerprints = {
            name: _fingerprint(torch, value)
            for name, value in zip(OUTPUT_NAMES, outputs)
        }
        snapshot = _cpu_snapshot(outputs)
        _stage("snapshot_done", repeat=repeat)
        repeat_summaries.append({
            "repeat": repeat,
            "attn_out_data_ptr": outputs[0].data_ptr(),
            "attn_out_tail": _tail_sample(torch, outputs[0]),
        })
        if baseline is not None:
            repeat_equal, equal_by_output, differences = _compare_snapshots(
                torch, snapshot, baseline, OUTPUT_NAMES, repeat
            )
            deterministic = (
                repeat_equal if deterministic is None
                else deterministic and repeat_equal
            )
            if deterministic_by_output is None:
                deterministic_by_output = equal_by_output
            else:
                deterministic_by_output = {
                    name: deterministic_by_output[name] and is_equal
                    for name, is_equal in equal_by_output.items()
                }
            binary_differences.extend(differences)
        elif args.repeats > 1:
            baseline = snapshot

    input_integrity = None
    input_differences = []
    if input_baseline is not None:
        input_integrity, _, input_differences = _compare_snapshots(
            torch,
            _cpu_snapshot(input_values),
            input_baseline,
            input_names,
            args.repeats,
        )
    try:
        device_name = torch.npu.get_device_name(args.device)
    except Exception as error:
        device_name = f"unavailable: {error!r}"
    try:
        triton_ascend_version = metadata.version("triton-ascend")
    except metadata.PackageNotFoundError:
        triton_ascend_version = "not-installed"
    memory_scale = 1024**3
    print(json.dumps({
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton_ascend": triton_ascend_version,
            "device_name": device_name,
            "fla_npu_module": fla_npu.__file__,
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        },
        "output_count": len(outputs),
        "elapsed_ms": (time.perf_counter() - started) * 1e3,
        "memory_allocated_gib": torch.npu.memory_allocated(device) / memory_scale,
        "memory_reserved_gib": torch.npu.memory_reserved(device) / memory_scale,
        "deterministic": deterministic,
        "deterministic_by_output": deterministic_by_output,
        "binary_differences": binary_differences,
        "input_integrity": input_integrity,
        "input_differences": input_differences,
        "repeat_summaries": repeat_summaries,
        "outputs": fingerprints,
    }))
    return 0 if deterministic is not False and all(
        value is None or value["finite"] for value in fingerprints.values()
    ) else 1


def _run_parent(args):
    if args.long_seq and args.bf16_gate_params:
        raise ValueError("--long-seq and --bf16-gate-params are mutually exclusive")
    long_seq = args.long_seq
    adapter = args.bf16_gate_params
    cases = LONG_CASES if long_seq else ADAPTER_CASES if adapter else SHORT_CASES
    heads = 96 if long_seq else args.heads
    timeout = args.timeout or (180 if long_seq else 30)
    repeats = 1 if long_seq else 5
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    print(json.dumps({
        "commit": commit, "device": args.device, "heads": heads,
        "long_seq": long_seq, "bf16_gate_params": adapter, "timeout": timeout,
    }))

    mismatch_found = False
    for name, tokens, final_state, saved in cases:
        command = [
            sys.executable, "-u", __file__, "--child",
            "--device", str(args.device), "--heads", str(heads),
            "--tokens", str(tokens), "--repeats", str(repeats),
        ]
        if long_seq:
            command.extend(("--constant-inputs", "--layout", "BSND"))
        if adapter:
            command.extend(
                ("--adapter", "--gate-params-bf16", "--layout", "BSND")
            )
        if final_state:
            command.append("--final-state")
        if saved:
            command.append("--saved")
        print(f"[RUN] {name}", flush=True)
        try:
            child_env = {**os.environ, "ASCEND_LAUNCH_BLOCKING": "1"}
            if adapter:
                child_env["FLA_NPU_KDA_ADAPTER_DEBUG_SYNC"] = "1"
            result = subprocess.run(
                command,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or ""
            if isinstance(captured, bytes):
                captured = captured.decode("utf-8", errors="replace")
            print(captured, end="")
            print(f"[TIMEOUT] {name} after {timeout}s; stop and reset the device")
            return 124
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, end="", file=sys.stderr)
            print(f"[FAIL] {name}: returncode={result.returncode}")
            if '"deterministic": false' in result.stdout:
                mismatch_found = True
                print(f"[CONTINUE] {name}: collect remaining tail diagnostics")
                continue
            return result.returncode
        print(f"[PASS] {name}")
    return 1 if mismatch_found else 0


if __name__ == "__main__":
    parsed = _parse_args()
    raise SystemExit(_run_child(parsed) if parsed.child else _run_parent(parsed))
