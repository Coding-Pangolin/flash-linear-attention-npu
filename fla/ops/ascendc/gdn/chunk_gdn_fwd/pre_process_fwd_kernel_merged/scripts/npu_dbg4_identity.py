"""调试分支专用：核对 kernel 里的 m 初值是否为严格的 128x128 单位阵。"""

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
    print("单位阵转储: max_abs=%.3e  非对角最大=%.3e" % (d.max().item(), (d - d.diag().diag()).max().item()))
    diag = dump.diag()
    print("对角前 4:", [round(x, 3) for x in diag[:4].tolist()],
          " 对角 92..100:", [round(x, 3) for x in diag[92:101].tolist()])
    print("对角不为 1 的行号:", [i for i in range(K) if abs(diag[i].item() - 1.0) > 1e-6][:20])
    offdiag_bad = (dump - dump.diag().diag()).abs().max().item()
    print("非对角元素最大绝对值:", offdiag_bad)
    # 每 16 列的对角正确个数
    ok = torch.tensor([abs(diag[i].item() - 1.0) < 1e-6 for i in range(K)])
    print("每 16 列对角正确个数:", [int(ok[i:i + 16].sum().item()) for i in range(0, K, 16)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
