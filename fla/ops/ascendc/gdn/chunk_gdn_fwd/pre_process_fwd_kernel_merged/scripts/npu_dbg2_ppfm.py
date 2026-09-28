"""定位 NPU 输出里"整片为 0"的分布（不依赖标杆）。"""

from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    op_dir = os.environ.get(
        "PPFM_OP_DIR",
        "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
        "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged")
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    torch.manual_seed(0)
    T, HK, HV, K, V = 64, 1, 1, 128, 128
    dev = "npu:0"
    k = torch.randn(T, HK, K).to(torch.bfloat16)
    v = torch.randn(T, HV, V).to(torch.bfloat16)
    w = torch.randn(T, HV, K).to(torch.bfloat16)

    def run():
        out = npu_ppfm(
            k.movedim(1, 0).unsqueeze(0).to(dev),
            w.movedim(1, 0).unsqueeze(0).to(dev),
            v.movedim(1, 0).unsqueeze(0).to(dev),
            g=torch.zeros(1, HV, T, dtype=torch.float32, device=dev),
            cu_seqlens=[0, T], chunk_size=64,
        ).cpu()[0, 0]
        torch.npu.synchronize()
        return out

    a = run()
    b = run()
    print("两次调用完全相同:", bool(torch.equal(a, b)))
    print("两次差 max_abs:", (a - b).abs().max().item())

    for tag, part in (("h", a[:, :V]), ("m", a[:, V:])):
        zero_row = (part.abs() < 1e-12).all(dim=1).nonzero().flatten().tolist()
        zero_col = (part.abs() < 1e-12).all(dim=0).nonzero().flatten().tolist()
        print(f"{tag}: 全零行 {len(zero_row)} 个 {zero_row[:8]} | 全零列 {len(zero_col)} 个 {zero_col[:8]}")
        print(f"{tag}: 按列的非零个数 min/max = "
              f"{int((part.abs() > 1e-12).sum(0).min())}/{int((part.abs() > 1e-12).sum(0).max())}")
    # 用"非零列数随 t 变化"看是否与列块有关
    nz = (a.abs() > 1e-12).sum(0)
    print("整行非零个数按列（每 16 列一组的均值）:",
          [int(nz[i:i + 16].float().mean().item()) for i in range(0, V + K, 16)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
