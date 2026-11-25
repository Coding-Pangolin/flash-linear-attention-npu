#!/usr/bin/env python3
"""Native FP32 communication gate; run with torchrun before whole-forward MC2.

All ranks check exact gather patterns and the ordered exclusive prefix. This
does not certify throughput or communication/computation overlap.
"""
from datetime import timedelta
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


def resource_snapshot(torch, device, local_rank):
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    physical = visible[local_rank] if visible != [""] else str(local_rank)
    result = {"torch_allocated_bytes": torch.npu.memory_allocated(device),
              "torch_reserved_bytes": torch.npu.memory_reserved(device)}
    try:
        p = subprocess.run(["npu-smi", "info", "-t", "proc-mem", "-i", physical],
                           capture_output=True, text=True, check=True, timeout=20)
        result["driver_process_report"] = p.stdout
        match = re.search(rf"Process id\s*:\s*{os.getpid()}\b.*?Process memory\(MB\)\s*:\s*(\d+)",
                          p.stdout, flags=re.I | re.S)
        result["driver_process_memory_mib"] = int(match.group(1)) if match else None
    except (subprocess.SubprocessError, OSError) as exc:
        result["driver_query_error"] = str(exc)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--negative", action="store_true", help="check local guards before data collectives")
    p.add_argument("--resources", action="store_true", help="separate diagnostic allocator/driver snapshots")
    p.add_argument("--progress-probe", action="store_true", help="read cycles from the opt-in A5 probe build")
    p.add_argument("--probe-rounds", choices=(1, 2), type=int, default=1)
    p.add_argument("--early-merge", action="store_true", help="read the CP2 early-prefix probe (build mode 3)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.heads <= 16 or a.repeats < 1:
        p.error("heads must be 1..16; repeats positive")
    import torch
    import torch_npu
    import torch.distributed as dist
    from fla_npu.ops.ascendc import kcp_all_gather_merge
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(seconds=60))
    group = dist.group.WORLD
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if not 2 <= size <= 8:
        raise ValueError("native gate requires CP2..8")
    if a.progress_probe and size != 2:
        raise ValueError("the A5 progress probe requires CP2")
    if a.probe_rounds == 2 and (not a.progress_probe or a.heads < 2):
        raise ValueError("two probe rounds require --progress-probe and at least two heads")
    if a.early_merge and (not a.progress_probe or a.probe_rounds != 2):
        raise ValueError("--early-merge requires --progress-probe --probe-rounds 2")
    # Initialize the process group's HCCL communicator before requesting its
    # name. The native op receives the same communicator, not a guessed name.
    metadata = [None] * size
    dist.all_gather_object(
        metadata, (a.heads, a.repeats, a.progress_probe, a.probe_rounds, a.early_merge), group=group
    )
    if any(item != metadata[0] for item in metadata):
        raise ValueError("rank metadata differs")
    device = torch.device(f"npu:{local_rank}")
    backend = group._get_backend(device)
    name = backend.get_hccl_comm_name(dist.get_rank())
    path = Path(__file__).resolve().parents[2] / "tests/atk/chunk_kda_cp_fwd/reference.py"
    spec = importlib.util.spec_from_file_location("kcp_reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    negative_checks = []
    if a.negative:
        summary = torch.zeros(a.heads, 128, 256, device=device)
        h = torch.zeros(a.heads, 128, 128, device=device)
        tests = (
            ("dtype", lambda: kcp_all_gather_merge(summary.bfloat16(), h, name, size, rank), "FP32"),
            ("h_shape", lambda: kcp_all_gather_merge(summary, h[..., :127].contiguous(), name, size, rank), "h.shape"),
            ("rank_range", lambda: kcp_all_gather_merge(summary, h, name, size, size), "0<=rank<P"),
            ("empty_group", lambda: kcp_all_gather_merge(summary, h, "", size, rank), "HCCL group name"),
        )
        for label, call, message in tests:
            try:
                call()
            except (ValueError, RuntimeError) as exc:
                if message not in str(exc):
                    raise AssertionError(f"{label}: missing parameter-specific error: {exc}") from exc
                negative_checks.append(label)
            else:
                raise AssertionError(f"{label}: invalid parameters reached execution")
        torch.npu.synchronize()
        dist.barrier(group)
    checks = []
    resources = {}
    for iteration in range(a.repeats):
        patterns = torch.zeros(size, a.heads, 128, 256)
        for source in range(size):
            for head in range(a.heads):
                # Exact binary fractions avoid confusing transport with GEMM
                # rounding. Vary by rank, head, coordinate and invocation.
                patterns[source, head, :, :128] = (
                    3 * torch.arange(128)[:, None] + 5 * torch.arange(128)[None, :]
                    + 4 * source + head + iteration) / 4096
                # Asymmetric E plus alternating upper/lower M catches an
                # accidental transpose, H@M, or reversed matrix composition.
                off_diagonal = torch.diag(torch.full((127,), 1 / 64),
                                          diagonal=1 if source % 2 == 0 else -1)
                patterns[source, head, :, 128:] = torch.eye(128) * (0.5 + source / 16) + off_diagonal
        if a.early_merge:
            # Probe mode 3 reserves the first 64 bytes of the rank 1 prefix
            # for markers. Their fixture values are zero and restored below.
            patterns[0, 0, 0, :16] = 0
        summary = patterns[rank].to(device)
        h = torch.zeros(a.heads, 128, 128, device=device)
        torch.npu.synchronize()
        if a.resources and iteration == 0:
            torch.npu.reset_peak_memory_stats(device)
            resources["before"] = resource_snapshot(torch, device, local_rank)
        print(json.dumps({"rank": rank, "iteration": iteration,
                          "phase": "enter_native_kernel"}), flush=True)
        with torch.no_grad():
            actual_h, gathered = kcp_all_gather_merge(summary, h, name, size, rank)
        torch.npu.synchronize()
        if a.resources and iteration == 0:
            resources["after"] = resource_snapshot(torch, device, local_rank)
            resources["torch_peak_allocated_bytes"] = torch.npu.max_memory_allocated(device)
            resources["gathered_output_bytes"] = gathered.numel() * gathered.element_size()
            resources["is_complete_whole_forward_memory_accounting"] = False
        print(json.dumps({"rank": rank, "iteration": iteration,
                          "phase": "native_kernel_completed"}), flush=True)
        torch.testing.assert_close(gathered.cpu(), patterns, rtol=0, atol=0)
        if actual_h.data_ptr() != h.data_ptr():
            raise AssertionError("native h output must alias its in-place input")
        h_cpu = actual_h.cpu()
        record = {"iteration": iteration, "exact_gather": True, "ordered_prefix": True}
        if a.early_merge and rank == 1:
            markers = h_cpu.flatten().view(torch.int64)[:8]
            submit0, ready0, submit1, wait1, merge_start, merge_end, padding0, padding1 = markers.tolist()
            if not (0 < submit0 < ready0 <= submit1 <= merge_start < merge_end <= wait1
                    and padding0 == padding1 == 0):
                raise AssertionError(f"unexpected early-merge markers: {markers.tolist()}")
            record.update(
                submit0=submit0, ready0=ready0, submit1=submit1, wait1_return=wait1,
                merge0_start=merge_start, merge0_end=merge_end,
                merge0_before_wait1=True, diagnostic_prefix_words=16,
            )
            markers.zero_()
        if a.progress_probe and not a.early_merge and rank == 0:
            markers = h_cpu.flatten().view(torch.int64)[:8]
            submit, ready, submit1, ready1, compute_start, compute_end, result, _ = markers.tolist()
            if not (0 < submit < ready and 0 < compute_start < compute_end and 1_400_000 < result < 1_600_000):
                raise AssertionError(f"unexpected A5 progress markers: {markers.tolist()}")
            overlap = max(0, min(ready, compute_end) - max(submit, compute_start))
            record.update(submit=submit, ready=ready, compute_start=compute_start,
                          compute_end=compute_end, wait_overlap_cycles=overlap)
            if a.probe_rounds == 2:
                if not ready <= submit1 < ready1:
                    raise AssertionError(f"unexpected second round markers: {markers.tolist()}")
                record.update(submit1=submit1, ready1=ready1,
                              wait1_overlap_cycles=max(0, min(ready1, compute_end) - max(submit1, compute_start)))
            elif submit1 or ready1:
                raise AssertionError(f"expected one A5 probe round: {markers.tolist()}")
            markers.zero_()
        torch.testing.assert_close(h_cpu.double(), ref.merge_reference(patterns, rank),
                                   rtol=1e-5, atol=1e-6)
        checks.append(record)
    all_checks = [None] * size
    all_resources = [None] * size
    dist.all_gather_object(all_checks, checks, group=group)
    dist.all_gather_object(all_resources, resources, group=group)
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "native_gate.json").write_text(json.dumps({
            "world": size, "heads": a.heads, "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "device": torch.npu.get_device_name(device),
            "checks": all_checks, "native_mc2": True, "overlap_validated": False,
            "wait_overlap_observed": any(item.get("wait_overlap_cycles", 0) > 0 for item in all_checks[0]),
            "early_merge": a.early_merge,
            "negative_checks": negative_checks,
            "resources": all_resources,
            "communication_environment": {key: os.environ.get(key) for key in (
                "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "PYTORCH_NPU_ALLOC_CONF")},
        }, indent=2), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
