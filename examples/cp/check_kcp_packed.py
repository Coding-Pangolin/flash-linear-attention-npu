#!/usr/bin/env python3
"""Compare A5 packed KDA CP against the independent FP64 oracle."""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--backend", choices=("staged", "fused"), default="staged")
    parser.add_argument("--case", action="append",
                        choices=("long", "aligned", "packed", "different_first_and_last"))
    parser.add_argument("--auto-head-group", action="store_true")
    parser.add_argument("--no-final-state", action="store_true")
    parser.add_argument("--weak-gate", action="store_true",
                        help="retain upstream state to exercise the multi-rank prefix")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 4 or args.heads < 2 or args.repeats < 1:
        parser.error("use at least 4 local tokens, 2 heads and 1 repetition")

    import torch
    import torch.distributed as dist
    import torch_npu  # Initializes the CANN runtime before the stable operator wrappers.
    from fla_npu.ops.ascendc import chunk_kda_cp_fwd, create_kda_cp_plan

    packed = load("fla_npu.ops.ascendc._kcp_packed",
                  "torch_custom/fla_npu/fla_npu/ops/ascendc/_kcp_packed.py")
    ref = load("kcp_packed_reference", "tests/atk/chunk_kda_cp_fwd/reference.py")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch.set_num_threads(1)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl", timeout=timedelta(seconds=60))
    try:
        group = dist.group.WORLD
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        tokens, heads = args.tokens, args.heads
        total = tokens * world
        fixtures = {
            "long": [0, total],
            "aligned": list(range(0, total + 1, tokens)),
            "packed": [0, *range(tokens // 2, total, tokens), total],
        }
        if world >= 4:
            fixtures["different_first_and_last"] = [
                0, tokens + tokens // 4, tokens + tokens // 2, 2 * tokens + tokens // 3, total,
            ]
        if args.case:
            fixtures = {case: fixtures[case] for case in args.case}
        generator = torch.Generator().manual_seed(912)
        shape = (1, heads, total, 128)
        q, k, v = (torch.randn(shape, generator=generator).bfloat16() for _ in range(3))
        g = torch.randn(shape, generator=generator)
        if args.weak_gate:
            g = g * 0.35 - 6.0
        beta = torch.randn(shape[:-1], generator=generator)
        a_log = 0.1 * torch.randn(heads, generator=generator)
        bias = 0.1 * torch.randn((heads, 128), generator=generator)
        start, end = rank * tokens, (rank + 1) * tokens
        local_inputs = [x[:, :, start:end].contiguous().to(device) for x in (q, k, v, g, beta)]
        device_a, device_bias = a_log.to(device), bias.to(device)
        records = []
        for case, cu in fixtures.items():
            expected_o, expected_h = ref.packed_shard_reference(
                q, k, v, g, beta, A_log=a_log, dt_bias=bias, cu_seqlens=cu, start=start, end=end)
            widths = (None,) if args.auto_head_group else (heads, (heads + 1) // 2)
            for width in widths:
                plan = packed.build_packed_plan(tokens, heads, world, 27, cu_seqlens=cu,
                                                head_group_size=width)
                print(json.dumps({"rank": rank, "case": case, "width": plan.head_group_size, "phase": "start"}),
                      flush=True)
                options = dict(A_log=device_a, dt_bias=device_bias, cp_group=group,
                               output_final_state=not args.no_final_state)
                if args.backend == "fused":
                    options.update(backend="fused", backward_mode="none")
                    native = create_kda_cp_plan(*local_inputs, **options, cu_seqlens=cu, head_group_size=width)
                first_out = first_state = None
                for iteration in range(args.repeats):
                    out = (chunk_kda_cp_fwd(*local_inputs, **options, plan=native) if args.backend == "fused"
                           else packed.packed_forward_staged(*local_inputs, **options, plan=plan))
                    got_o = out.output.cpu().double()
                    torch.testing.assert_close(got_o, expected_o, atol=0.005, rtol=0.005)
                    if args.backend == "fused" and args.repeats > 1:
                        if first_out is None:
                            first_out = got_o
                        else:
                            torch.testing.assert_close(got_o, first_out, atol=0, rtol=0)
                    state_error = None
                    if args.no_final_state:
                        if out.final_state is not None or out.cache is not None:
                            raise AssertionError("state and cache must be absent")
                    else:
                        got_h = out.final_state.cpu().double()
                        torch.testing.assert_close(got_h, expected_h, atol=0.02, rtol=0.02)
                        if args.backend == "fused" and args.repeats > 1:
                            if first_state is None:
                                first_state = got_h
                            else:
                                torch.testing.assert_close(got_h, first_state, atol=0, rtol=0)
                        state_error = (got_h - expected_h).abs().max().item()
                    records.append({
                        "rank": rank, "case": case, "width": plan.head_group_size, "rounds": plan.rounds,
                        "iteration": iteration, "sequences": plan.shards[rank].sequences,
                        "output_max_abs": (got_o - expected_o).abs().max().item(),
                        "state_max_abs": state_error,
                    })
        all_records = [None] * world
        dist.all_gather_object(all_records, records, group=group)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "backend": "packed_" + args.backend, "cp": world, "tokens": tokens, "heads": heads,
                "device": torch.npu.get_device_name(device), "weak_gate": args.weak_gate, "torch": torch.__version__,
                "torch_npu": torch_npu.__version__, "records": all_records,
            }, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
