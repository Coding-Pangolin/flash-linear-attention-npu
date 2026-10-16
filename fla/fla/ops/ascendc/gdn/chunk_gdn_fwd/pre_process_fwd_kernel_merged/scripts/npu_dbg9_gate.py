"""GDN 门控路径定位：单 chunk（T=64）+ 非零 gate，核对 h 的闭式解。"""

from __future__ import annotations

import torch


def run(T, gate_scale, seed=0):
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm
    from reference import pre_process_fwd_kernel_merged as ref_ppfm

    torch.manual_seed(seed)
    HK = HV = K = V = 1
    K, V = 128, 128
    HK = HV = 1
    dev = "npu:0"
    k = torch.randn(T, HK, K).to(torch.bfloat16)
    v = torch.randn(T, HV, V).to(torch.bfloat16)
    w = torch.randn(T, HV, K).to(torch.bfloat16)
    g = (-torch.rand(T, HV) * gate_scale).to(torch.float32)

    ref = ref_ppfm(k, v, w, g=g, chunk_size=64, cu_seqlens=[0, T])[0]
    got = npu_ppfm(
        k.movedim(1, 0).unsqueeze(0).to(dev),
        w.movedim(1, 0).unsqueeze(0).to(dev),
        v.movedim(1, 0).unsqueeze(0).to(dev),
        g=g.movedim(1, 0).unsqueeze(0).to(dev),
        cu_seqlens=[0, T], chunk_size=64,
    ).cpu()[0, 0]
    torch.npu.synchronize()

    # 闭式解（单 chunk、h 初值 0）：L_t = bf16(2^(g_last-g_t) k_t)，v_new = v * 2^(g_last-g_t)
    dg = torch.exp2(g[-1] - g[:, 0])                       # [T]
    L = (k[:, 0].float() * dg[:, None]).to(torch.bfloat16).float()
    vn = (v[:, 0].float() * dg[:, None]).to(torch.bfloat16).float()
    h_closed = L.t() @ vn
    kw_closed = L.t() @ w[:, 0].float()
    m_closed = torch.eye(K) * float(torch.exp2(g[-1, 0])) - kw_closed

    for tag, dump, want in (("h(NPU vs 标杆)", got[:, :V], ref[:, :V]),
                            ("m(NPU vs 标杆)", got[:, V:], ref[:, V:]),
                            ("h(NPU vs 闭式)", got[:, :V], h_closed)):
        d = (dump - want).abs()
        print(f"  {tag:8s} max_abs={d.max().item():.3e} mean={d.mean().item():.3e}")
    print(f"  |h|max: NPU={got[:, :V].abs().max().item():.3e}  标杆={ref[:, :V].abs().max().item():.3e}")
    return


def main() -> int:
    import sys
    sys.path.insert(0, "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
                       "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
    for T in (64, 128, 256):
        print(f"=== T={T}（gate ∈ (-1,0)）===")
        run(T, gate_scale=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
