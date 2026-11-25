#!/usr/bin/env python3
"""Control experiment using CANN's stock AllGatherMatmulV2, without fla_npu.

This distinguishes native resource-initialization failures from KCP arithmetic.
The BF16 control does not prove that FP32 transport is supported.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("ccu", "ai_cpu"), default="ccu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch_npu
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=60))
    group = dist.group.WORLD
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device(f"npu:{local_rank}")
    probe = torch.tensor([rank], device=device)
    received = [torch.empty_like(probe) for _ in range(size)]
    dist.all_gather(received, probe, group=group)
    name = group._get_backend(device).get_hccl_comm_name(dist.get_rank())
    x = torch.full((128, 256), rank + 1, dtype=torch.bfloat16, device=device)
    w = torch.eye(256, dtype=torch.bfloat16, device=device)[:, :128].contiguous()
    print(json.dumps({"rank": rank, "phase": "enter_stock_mc2", "mode": a.mode}), flush=True)
    out, gathered = torch_npu.npu_all_gather_base_mm(x, w, name, size, comm_mode=a.mode)
    torch.npu.synchronize()
    expected = torch.cat([torch.full((128, 256), i+1, dtype=torch.bfloat16) for i in range(size)])
    torch.testing.assert_close(gathered.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), expected[:, :128], rtol=0, atol=0)
    print(json.dumps({"rank": rank, "phase": "stock_mc2_passed"}), flush=True)
    dist.barrier(group)
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output/"stock_gate.json").write_text(json.dumps({
            "status": "passed", "world": size, "dtype": "BF16", "mode": a.mode,
            "device": torch.npu.get_device_name(device), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "environment": {key: os.environ.get(key) for key in (
                "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "PYTORCH_NPU_ALLOC_CONF")},
        }, indent=2), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
