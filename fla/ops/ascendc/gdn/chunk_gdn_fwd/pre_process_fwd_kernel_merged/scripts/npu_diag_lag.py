"""判据：出错的 h 是否正好"落后一个 chunk"（与参考在 [0, T-BT] 的末状态逐位比对）。"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/tmp")
from npu_one_case import build_case, compare, to_npu_bnsd  # noqa: E402

REF = ("/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
       "chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
sys.path.insert(0, REF)


def main() -> int:
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm
    from reference import pre_process_fwd_kernel_merged as ref_ppfm

    dev = "npu:0"
    T, K, V, BT = 256, 128, 128, 64
    for rep in range(6):
        case = build_case(1000, T, 2, 2, K, V, "gdn", BT=BT)
        npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in case.items()}
        got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
                       cu_seqlens=[0, T], chunk_size=BT, g=npu_in["g"].to(dev), gk=None)
        torch.npu.synchronize()
        got = got.cpu()[0]

        kw = dict(g=case["g"], gk=case["gk"], chunk_size=BT)
        want = ref_ppfm(case["k"], case["v"], case["w"], cu_seqlens=[0, T], **kw)
        lag = ref_ppfm(case["k"], case["v"], case["w"], cu_seqlens=[0, T - BT], **kw)
        lg2 = ref_ppfm(case["k"], case["v"], case["w"], cu_seqlens=[0, T - 2 * BT], **kw)

        out = [f"rep{rep}"]
        for hv in range(2):
            g_h = got[hv, :, :V]
            s_ok = compare(g_h, want[hv, :, :V])
            s_l1 = compare(g_h, lag[hv, :, :V])
            s_l2 = compare(g_h, lg2[hv, :, :V])
            s_okm = compare(got[hv, :, V:], want[hv, :, V:])
            s_l1m = compare(got[hv, :, V:], lag[hv, :, V:])
            out.append(f"[h{hv}] vsCur:{s_ok['matched']:.3f}/{s_ok['max_abs']:.1e} "
                       f"vsLag1:{s_l1['matched']:.3f}/{s_l1['max_abs']:.1e} "
                       f"vsLag2:{s_l2['matched']:.3f}/{s_l2['max_abs']:.1e} || "
                       f"m vsCur:{s_okm['matched']:.3f}/{s_okm['max_abs']:.1e} "
                       f"vsLag1:{s_l1m['matched']:.3f}/{s_l1m['max_abs']:.1e}")
        print("  ".join(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
