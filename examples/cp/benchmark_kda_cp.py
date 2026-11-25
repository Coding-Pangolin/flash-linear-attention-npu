#!/usr/bin/env python3
"""All-rank KDA CP accuracy and joined-stream A/B benchmark.

torchrun --standalone --nproc_per_node=2 examples/cp/benchmark_kda_cp.py \
  --tokens 65 --heads 2 --accuracy --warmup 1 --iterations 2 --windows 1
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=2048, help="tokens per CP rank")
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--cp-groups", type=int, default=1,
                   help="interleaved independent CP groups, exercising non-WORLD communicators")
    p.add_argument("--mode", choices=("save", "none"), default="none")
    p.add_argument("--backends", default="staged,pipeline")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--backward", action="store_true")
    p.add_argument("--random-do", action="store_true", help="nonuniform BF16 output gradients for explicit-backward validation")
    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--no-cp-reference", action="store_true",
                   help="separate accuracy-only run: compare the staged CP baseline to head-batched no-CP prefixes")
    p.add_argument("--reference-heads", type=int, default=2)
    p.add_argument("--negative-preflight", action="store_true")
    p.add_argument("--resources", action="store_true",
                   help="diagnostic cold-call snapshots before accuracy/timing; use one backend per fresh process")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--vary-parameters", action="store_true",
                   help="accuracy control with distinct nonzero A_log/bias across heads and coordinates")
    p.add_argument("--flat-bias", action="store_true")
    p.add_argument("--no-final-state", action="store_true")
    p.add_argument("--stress-repeats", type=int, default=0,
                   help="alternate value signs to detect stale cross-stream buffers")
    p.add_argument("--output", type=Path, default=Path("outputs/kcp"))
    a = p.parse_args()
    if min(a.tokens, a.heads, a.group_size, a.reference_heads, a.iterations, a.windows) < 1 or min(a.warmup, a.stress_repeats) < 0:
        p.error("dimensions, iterations and windows must be positive; warmup nonnegative")
    if a.backward and a.mode != "save":
        p.error("--backward requires --mode save")
    if a.random_do and not a.backward:
        p.error("--random-do requires --backward")
    if a.accuracy_only and not (a.accuracy or a.no_cp_reference):
        p.error("--accuracy-only requires --accuracy or --no-cp-reference")
    if a.no_cp_reference and (not a.accuracy_only or not a.backends.startswith("staged")):
        p.error("--no-cp-reference requires --accuracy-only and staged as the first backend")
    if a.accuracy and a.tokens * a.heads * int(os.environ.get("WORLD_SIZE", "1")) > 8192:
        p.error("CPU reference limited to global tokens*heads <=8192; use smaller accuracy cases")
    return a


def load_reference():
    path = Path(__file__).resolve().parents[2] / "tests/atk/chunk_kda_cp_fwd/reference.py"
    spec = importlib.util.spec_from_file_location("kcp_cpu_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


def main():
    a = args()
    import torch
    import torch_npu
    import torch.distributed as dist
    from fla_npu.ops.ascendc import (chunk_kda_cp_fwd, chunk_kda_cp_bwd, create_kda_cp_plan)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    global_rank, global_world = dist.get_rank(), dist.get_world_size()
    if a.cp_groups < 1 or global_world % a.cp_groups:
        raise ValueError("--cp-groups must divide the global process count")
    members = [list(range(i, global_world, a.cp_groups)) for i in range(a.cp_groups)]
    if a.cp_groups == 1:
        group = dist.group.WORLD  # Deliberate benchmark CP group, not library default.
    else:
        groups = [dist.new_group(ranks=ranks, backend="hccl") for ranks in members]
        group = groups[global_rank % a.cp_groups]
        a.output = a.output / f"group_{global_rank % a.cp_groups}"
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    # Every process creates singleton groups in the same global order. The
    # independent control performs ordinary no-CP compute on its own prefix.
    singleton = None
    if a.no_cp_reference:
        singletons = [dist.new_group(ranks=[r], backend="hccl") for r in range(global_world)]
        singleton = singletons[global_rank]
    device = torch.device(f"npu:{local_rank}")
    generator = torch.Generator().manual_seed(a.seed + global_rank)
    shape = (1, a.heads, a.tokens, 128)
    cpu = {name: torch.randn(shape, generator=generator).bfloat16() for name in ("q", "k", "v")}
    cpu["v"] *= 0.5
    cpu["g"] = torch.randn(shape, generator=generator) * 0.1 - 5.0
    cpu["beta"] = torch.randn(shape[:-1], generator=generator)
    cpu["A_log"] = torch.zeros(a.heads)
    cpu["dt_bias"] = torch.zeros(a.heads, 128)
    if a.vary_parameters:
        cpu["A_log"] = torch.linspace(-0.5, 0.5, a.heads)
        cpu["dt_bias"] = torch.linspace(-0.1, 0.1, a.heads * 128).reshape(a.heads, 128)
    x = {name: tensor.to(device) for name, tensor in cpu.items()}
    if a.no_bias:
        x["dt_bias"] = None  # CPU oracle's zero bias is mathematically equivalent.
        cpu["dt_bias"].zero_()
    elif a.flat_bias:
        x["dt_bias"] = x["dt_bias"].flatten()
    state_requested = not a.no_final_state
    backends = a.backends.split(",")
    if len(set(backends)) != len(backends):
        raise ValueError("duplicate backends")
    plans = {backend: create_kda_cp_plan(
        **x, cp_group=group, backend=backend, backward_mode=a.mode,
        head_group_size=a.group_size, output_final_state=state_requested) for backend in backends}
    negative_checks = []
    if a.negative_preflight:
        if world < 2:
            raise ValueError("negative collective preflight needs CP>=2")
        for label in ("mismatched_group_width", "one_rank_wrong_dtype", "one_rank_requires_grad"):
            bad = dict(x)
            width = a.group_size
            if label == "mismatched_group_width":
                width = 1 if rank == 0 else 2
            elif label == "one_rank_wrong_dtype" and rank == world - 1:
                bad["q"] = bad["q"].float()
            elif label == "one_rank_requires_grad" and rank == world - 1:
                bad["q"] = bad["q"].detach().requires_grad_()
            try:
                create_kda_cp_plan(**bad, cp_group=group, backward_mode=a.mode,
                                   backend="pipeline", head_group_size=width, output_final_state=state_requested)
            except ValueError as exc:
                expected_error = {"mismatched_group_width": "mismatched input metadata or options",
                    "one_rank_wrong_dtype": "q has unsupported dtype",
                    "one_rank_requires_grad": "detached inputs"}[label]
                if "KDA CP preflight failed" not in str(exc) or expected_error not in str(exc):
                    raise
                negative_checks.append(label)
            else:
                raise AssertionError(f"{label}: all ranks must reject before data collectives")
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = a.output / f"measurements_rank{rank}.jsonl"
    existing = [None] * world
    dist.all_gather_object(existing, checkpoint.exists() or (a.output/"manifest.json").exists()
                           or (a.output/"resources.json").exists(), group=group)
    if any(existing):
        raise FileExistsError("use a fresh output directory; a checkpoint or result already exists")

    def run(backend):
        return chunk_kda_cp_fwd(**x, cp_group=group, backend=backend,
                                backward_mode=a.mode, head_group_size=a.group_size,
                                output_final_state=state_requested, plan=plans[backend])

    resource_checks = {}
    if a.resources:
        # Reuse the native gate's own-PID driver query, outside every event.
        from check_kcp_native import resource_snapshot
        if len(backends) != 1:
            raise ValueError("--resources requires one backend and a fresh process for attributable cold allocation")
        backend = backends[0]
        torch.npu.synchronize()
        before = resource_snapshot(torch, device, local_rank)
        torch.npu.reset_peak_memory_stats(device)
        with torch.no_grad():
            cold = run(backend)
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated(device)
        after = resource_snapshot(torch, device, local_rank)
        input_storage = {t.untyped_storage().data_ptr() for t in x.values() if t is not None}
        returned = [cold.output, cold.final_state]
        if cold.cache is not None:
            returned.extend(t for part in cold.cache.groups for t in part.tensors.values())
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for t in returned if t is not None and t.untyped_storage().data_ptr() not in input_storage}
        resource_checks = dict(backend=backend, before=before, after=after,
            torch_peak_allocated_bytes=peak, newly_owned_result_storage_bytes=sum(storages.values()),
            torch_peak_minus_initial_bytes=peak-before["torch_allocated_bytes"],
            is_complete_temporary_memory_accounting=False)
        del cold
        torch.npu.synchronize()
        all_resources = [None] * world
        dist.all_gather_object(all_resources, resource_checks, group=group)
        if rank == 0:
            (a.output/"resources.json").write_text(json.dumps(all_resources, indent=2), encoding="utf-8")

    records, checks = [], {}
    do_cpu = (torch.randn((1,a.tokens,a.heads,128),
              generator=torch.Generator().manual_seed(a.seed+111111+global_rank)).bfloat16()
              if a.random_do else None)
    with torch.no_grad():
        baseline = run(backends[0])
        baseline_output = baseline.output.cpu()
        baseline_state = baseline.final_state.cpu() if state_requested else None
        if a.no_cp_reference:
            # Gather only a small head batch at a time: never materialize a
            # whole global-sequence, all-head copy on each device.
            for first in range(0, a.heads, a.reference_heads):
                last = min(first+a.reference_heads, a.heads)
                global_inputs = {}
                for name in ("q", "k", "v", "g", "beta"):
                    local = x[name][:, first:last].contiguous()
                    gathered = torch.empty((world, *local.shape[1:]), dtype=local.dtype, device=device)
                    dist.all_gather_into_tensor(gathered, local, group=group)
                    # [P,G,T,D] -> [1,G,P*T,D], preserving CP group rank order.
                    axes = (1,0,2,3) if local.ndim == 4 else (1,0,2)
                    dense = gathered.permute(axes).reshape(1,last-first,world*a.tokens,*local.shape[3:])
                    global_inputs[name] = dense[:, :, :(rank+1)*a.tokens].contiguous()
                bias = None if x["dt_bias"] is None else x["dt_bias"].reshape(a.heads,128)[first:last].clone()
                independent = chunk_kda_cp_fwd(**global_inputs, cp_group=singleton,
                    A_log=x["A_log"][first:last].clone(), dt_bias=bias, backend="staged",
                    backward_mode="none", output_final_state=state_requested)
                want = independent.output[:, rank*a.tokens:(rank+1)*a.tokens].cpu().float()
                torch.testing.assert_close(baseline_output[:,:,first:last].float(), want, rtol=0.005, atol=0.005)
                if state_requested:
                    torch.testing.assert_close(baseline_state[:,first:last].float(),
                        independent.final_state.cpu().float(), rtol=0.02, atol=0.02)
                print(json.dumps({"rank": rank, "phase": "no_cp_head_batch_checked",
                                  "first_head": first, "last_head": last}), flush=True)
                del independent, global_inputs, local, gathered, dense, want, bias
            torch.npu.synchronize()
        reference = load_reference()
        expected = expected_state = None
        if a.accuracy:
            inputs = [None] * world
            dist.all_gather_object(inputs, cpu, group=group)
            global_x = {n: torch.cat([item[n] for item in inputs], dim=2)
                        for n in ("q", "k", "v", "g", "beta")}
            prefix = {name: tensor[:, :, :(rank+1)*a.tokens] for name, tensor in global_x.items()}
            expected, expected_state = reference(**prefix, A_log=cpu["A_log"], dt_bias=cpu["dt_bias"])
            expected = expected[:, rank*a.tokens:(rank+1)*a.tokens]
            if a.random_do:
                all_do = [None] * world
                dist.all_gather_object(all_do, do_cpu, group=group)
                global_do = torch.cat(all_do, dim=1)
        for backend in backends:
            print(json.dumps({"rank": rank, "phase": "check_backend", "backend": backend}), flush=True)
            result = run(backend)
            actual = result.output.cpu()
            if actual.shape != baseline_output.shape or actual.dtype != torch.bfloat16:
                raise AssertionError("output must preserve BF16 BSND shape")
            # Compare every element, with the same thresholds, without several
            # simultaneous all-head FP32 CPU copies on large-sequence cases.
            for first in range(0, a.heads, a.reference_heads):
                last = min(first+a.reference_heads, a.heads)
                torch.testing.assert_close(actual[:,:,first:last].float(),
                    baseline_output[:,:,first:last].float(), rtol=0.005, atol=0.005)
            if state_requested:
                torch.testing.assert_close(result.final_state.cpu().float(), baseline_state.float(),
                                           rtol=0.02, atol=0.02)
            elif result.final_state is not None:
                raise AssertionError("final_state must be None when not requested")
            if expected is not None:
                torch.testing.assert_close(actual.double(), expected, rtol=0.005, atol=0.005)
                if state_requested:
                    torch.testing.assert_close(result.final_state.cpu().double(), expected_state, rtol=0.02, atol=0.02)
            checks[backend] = {"baseline_parity": True, "cpu_reference": expected is not None}
            if a.backward:
                do = do_cpu.to(device) if a.random_do else torch.ones_like(result.output)
                if a.negative_preflight:
                    for label in ("wrong_dtype", "wrong_device", "requires_grad"):
                        bad_do = do
                        if rank == world - 1:
                            bad_do = (do.float() if label == "wrong_dtype" else
                                      do.cpu() if label == "wrong_device" else
                                      do.detach().requires_grad_())
                        try:
                            chunk_kda_cp_bwd(bad_do, result.cache)
                        except ValueError as exc:
                            if "KDA CP backward preflight failed" not in str(exc):
                                raise
                        else:
                            raise AssertionError(f"backward {label}: all ranks must reject")
                        del bad_do
                        if "backward_" + label not in negative_checks:
                            negative_checks.append("backward_" + label)
                gradients = chunk_kda_cp_bwd(do, result.cache)
                checks[backend]["backward_finite"] = all(bool(t.isfinite().all()) for t in gradients.values())
                if not checks[backend]["backward_finite"]:
                    raise AssertionError("nonfinite gradients")
                if backend == backends[0]:
                    base_grad = {n: t.cpu() for n, t in gradients.items()}
                else:
                    for name, tensor in gradients.items():
                        torch.testing.assert_close(tensor.cpu().float(), base_grad[name].float(),
                                                   rtol=0.02, atol=0.02)
                if expected is not None:
                    leaves = {n: t.float().detach().requires_grad_() for n, t in global_x.items()}
                    aa = cpu["A_log"].clone().requires_grad_()
                    bb = cpu["dt_bias"].clone().requires_grad_()
                    with torch.enable_grad():
                        # Rebuild with grad enabled: the caller uses no_grad for NPU.
                        truth, _ = reference(**leaves, A_log=aa, dt_bias=bb)
                        loss = (truth * global_do.to(truth.dtype)).sum() if a.random_do else truth.sum()
                        loss.backward()
                    for name in ("q", "k", "v", "g", "beta"):
                        want = leaves[name].grad[:, :, rank*a.tokens:(rank+1)*a.tokens].transpose(1, 2)
                        torch.testing.assert_close(gradients["d"+name].cpu().float(), want.float(),
                                                   rtol=0.008 if name in ("q", "k", "v") else 0.02,
                                                   atol=0.008 if name in ("q", "k", "v") else 0.02)
                    for name, want in (("dA_log", aa.grad), ("ddt_bias", bb.grad)):
                        torch.testing.assert_close(gradients[name].cpu().float(), want.float(), rtol=0.02, atol=0.02)
                    checks[backend]["cpu_backward"] = True
                del gradients, do
            print(json.dumps({"rank": rank, "phase": "checked_backend", "backend": backend}), flush=True)
        for iteration in range(a.stress_repeats):
            x["v"].neg_()
            sign = -1 if iteration % 2 == 0 else 1
            for backend in backends:
                result = run(backend)
                torch.testing.assert_close(result.output.cpu().float(), baseline_output.float() * sign,
                                           rtol=0.005, atol=0.005)
                if state_requested:
                    torch.testing.assert_close(result.final_state.cpu().float(), baseline_state.float() * sign,
                                               rtol=0.02, atol=0.02)
        if a.stress_repeats % 2:
            x["v"].neg_()
        for check in checks.values():
            check["alternating_value_stress_repeats"] = a.stress_repeats
        del baseline, result
        torch.npu.synchronize()
        for window in range(0 if a.accuracy_only else a.windows):
            order = backends if window % 2 == 0 else list(reversed(backends))
            for backend in order:
                for _ in range(a.warmup):
                    result = run(backend)
                    del result
                torch.npu.synchronize()
                dist.barrier(group)
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats(device)
                resident = torch.npu.memory_allocated(device)
                events = []
                for iteration in range(a.iterations):
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    result = run(backend)
                    end.record()
                    del result
                    events.append((iteration, start, end))
                torch.npu.current_stream().synchronize()
                peak = torch.npu.max_memory_allocated(device)
                for iteration, start, end in events:
                    records.append(dict(window=window, rank=rank, iteration=iteration,
                                        backend=backend, duration_us=start.elapsed_time(end)*1000,
                                        resident_torch_allocated_bytes=resident,
                                        peak_torch_allocated_bytes=peak,
                                        incremental_torch_peak_bytes=peak-resident))
                # Preserve completed windows even if a later case times out.
                # Host I/O and progress messages are outside all timed events.
                with checkpoint.open("a", encoding="utf-8") as stream:
                    for row in records[-a.iterations:]:
                        stream.write(json.dumps(row) + "\n")
                print(json.dumps({"rank": rank, "phase": "window_completed", "window": window,
                                  "backend": backend, "peak_torch_allocated_bytes": peak}), flush=True)
        all_records = [None] * world
        all_checks = [None] * world
        dist.all_gather_object(all_records, records, group=group)
        dist.all_gather_object(all_checks, checks, group=group)
        if rank == 0:
            a.output.mkdir(parents=True, exist_ok=True)
            manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                            world=world, torch=torch.__version__, torch_npu=torch_npu.__version__,
                            global_world=global_world, group_members=members[global_rank % a.cp_groups],
                            baseline_backend=backends[0],
                            independent_no_cp_baseline_control=a.no_cp_reference,
                            device=torch.npu.get_device_name(device), host_arch=platform.machine(),
                            checks=all_checks, timing="NPU event joined makespan; not Task Duration",
                            negative_preflight_checks=negative_checks,
                            native_mc2="mc2" in backends, overlap_validated=False)
            manifest["communication_environment"] = {name: os.environ.get(name) for name in (
                "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "HCCL_EXEC_TIMEOUT",
                "HCCL_ALGO", "ASCEND_RT_VISIBLE_DEVICES")}
            from fla_npu.ops.ascendc import _kcp, _stable_hash
            manifest["orchestration_sha256"] = hashlib.sha256(Path(_kcp.__file__).read_bytes()).hexdigest()
            manifest["launcher_source_hash"] = _stable_hash.SOURCE_HASH
            (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (a.output/"measurements.json").write_text(json.dumps([r for rows in all_records for r in rows]), encoding="utf-8")
            if records:
                path = Path(__file__).resolve().parents[2] / "scripts/kcp_experiment.py"
                spec = importlib.util.spec_from_file_location("kcp_experiment", path)
                experiment = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(experiment)
                summary = experiment.aggregate_windows([r for rows in all_records for r in rows],
                    expected_ranks=world, expected_iterations=a.iterations, expected_windows=a.windows)
                (a.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(manifest))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
