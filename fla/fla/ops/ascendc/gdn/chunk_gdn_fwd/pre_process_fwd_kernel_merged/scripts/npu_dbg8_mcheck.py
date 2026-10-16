"""调试分支专用：核对进入 m 链之前 mF_ 是否仍为单位阵（检查 UB 是否被覆盖）。"""

from __future__ import annotations

import torch


def main() -> int:
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    torch.manual_seed(0)
    T, HK, HV, K, V = 64, 1, 1, 128, 128
    dev = "npu:0"
    k = torch.randn(T, HK, K).to(torch.bfloat16)
    v = torch.randn(T, HV, V).to(torch.bfloat16)
    w = torch.randn(T, HV, K).to(torch.bfloat16)
    got = npu_ppfm(
        k.movedim(1, 0).unsqueeze(0).to(dev),
        w.movedim(1, 0).unsqueeze(0).to(dev),
        v.movedim(1, 0).unsqueeze(0).to(dev),
        g=torch.zeros(1, HV, T, dtype=torch.float32, device=dev),
        cu_seqlens=[0, T], chunk_size=64,
    ).cpu()[0, 0]
    torch.npu.synchronize()

    ident = torch.eye(K)
    dump = got[:, :V]
    d = (dump - ident).abs()
    diag = dump.diag()
    bad = d > 1e-6
    print(f"mF_ 转储: max_abs={d.max().item():.3e}  错误元素={int(bad.sum())}/{bad.numel()}")
    print("对角前 4:", [round(x, 3) for x in diag[:4].tolist()])
    rows = bad.any(dim=1).nonzero().flatten().tolist()
    cols = bad.any(dim=0).nonzero().flatten().tolist()
    print("出错行数:", len(rows), "前几个:", rows[:8])
    print("出错列数:", len(cols), "前几个:", cols[:8])
    if rows and cols:
        r0, c0 = rows[0], cols[0]
        print(f"示例 dump[{r0},{c0}]={dump[r0, c0].item():.4f} 期望={ident[r0, c0].item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
