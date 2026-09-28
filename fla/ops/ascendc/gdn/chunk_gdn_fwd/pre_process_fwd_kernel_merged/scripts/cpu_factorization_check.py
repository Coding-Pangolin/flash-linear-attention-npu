"""CPU 侧验证：m_new = dc⊙m - Lᵀ(W m) 与标杆的 M@m 在 fp32 下的差异。

如果两者在 gdn-single 这类随机满幅 w 的用例上就能差出 ~1e-1 相对误差，
说明该用例本身病态（|m| 达到 1e7，链式乘积把 fp32 求和顺序的差异放大了），
而不是内核实现有 bug。
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
                   "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402


def merged_by_factorization(k, v, w, g, BT, T, K, V):
    """按内核的因式分解重算 m：T1 = W_c @ m; T2 = L^T @ T1; m = dc⊙m - T2。"""
    HV = w.shape[1]
    m = torch.eye(K, dtype=torch.float32).repeat(HV, 1, 1)
    for c in range(0, -(-T // BT)):
        lo, hi = c * BT, min((c + 1) * BT, T)
        kc = k[lo:hi].float()                      # [bt,HK,K]
        wc = w[lo:hi].float()                      # [bt,HV,K]
        gl = g[hi - 1]                             # [HV]
        gs = g[lo:hi]                              # [bt,HV]
        scale = torch.exp2(gl[:, None] - gs.permute(1, 0))   # [HV,bt]
        left = kc.permute(1, 0, 2) * scale.unsqueeze(-1)     # [HK,bt,K]
        left = left.to(torch.bfloat16).float()     # 契约：left 先降 bf16
        t1 = torch.einsum("thk,hkj->thj", wc, m)   # [bt,HV,K]
        t2 = torch.einsum("htk,thj->hkj", left, t1)
        m = torch.exp2(gl)[:, None, None] * m - t2
    return m


def main() -> int:
    torch.manual_seed(0)
    T, HK, HV, K, V, BT = 256, 2, 2, 128, 128, 64
    gg = torch.Generator().manual_seed(1000)
    k = torch.randn(T, HK, K, generator=gg).bfloat16()
    v = torch.randn(T, HV, V, generator=gg).bfloat16()
    w = torch.randn(T, HV, K, generator=gg).bfloat16()
    nblk = -(-T // BT)
    base = -0.013 / BT * (1 + torch.rand(nblk, BT, HV, generator=gg) * 0.5)
    g = base.cumsum(1).reshape(-1, HV)[:T].contiguous()

    ref = ref_ppfm(k, v, w, g=g, chunk_size=BT, cu_seqlens=[0, T])[0]
    alt = merged_by_factorization(k, v, w, g, BT, T, K, V)
    m_ref, m_alt = ref[:, V:], alt[0]      # ref 已按 [0] 取到单 head
    d = (m_alt - m_ref).abs().reshape(-1)
    mref = m_ref.abs().reshape(-1)
    limit = 1.5e-2 + 2e-3 * mref
    print(f"CPU 两种求和顺序: matched={(d <= limit).float().mean().item():.6f} "
          f"max_abs={d.max().item():.3e} |m|max={mref.max().item():.3e}")
    bad = d > limit
    if bad.any():
        print(f"   错元素 {int(bad.sum())} 个，相对误差中位数="
              f"{(d[bad] / mref[bad].clamp_min(1e-9)).median().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
