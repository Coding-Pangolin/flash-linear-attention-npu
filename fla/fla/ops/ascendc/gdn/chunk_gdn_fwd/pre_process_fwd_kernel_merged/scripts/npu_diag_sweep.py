"""同一进程内跑多个 gdn 定长配置，打印每个 hv 的 h/m 半边精度（定位"哪个核/哪个 task 坏"）。"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/tmp")
from npu_one_case import build_case, compare, to_npu_bnsd  # noqa: E402

REF_DIR = ("/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
           "chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
sys.path.insert(0, REF_DIR)


def one(dev, ref_ppfm, npu_ppfm, T, HK, HV, K, V, BT, cu, seed=1000, tag=""):
    case = build_case(seed, T, HK, HV, K, V, "gdn", BT=BT)
    npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in case.items()}
    got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
                   cu_seqlens=list(cu), chunk_size=BT, g=npu_in["g"].to(dev), gk=None)
    torch.npu.synchronize()
    got = got.cpu()
    want = []
    for bos, eos in zip(cu, cu[1:]):
        want.append(ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                             chunk_size=BT, cu_seqlens=[bos, eos]))
    want = torch.stack(want, dim=0)
    line = f"{tag:22s} T={T:4d} HK={HK} HV={HV} cu={cu}: "
    for n in range(got.shape[0]):
        for hv in range(HV):
            sh = compare(got[n, hv, :, :V], want[n, hv, :, :V])
            sm = compare(got[n, hv, :, V:], want[n, hv, :, V:])
            line += (f"[n{n}h{hv}] h:{sh['matched']:.3f}/{sh['max_abs']:.1e} "
                     f"m:{sm['matched']:.3f}/{sm['max_abs']:.1e}  ")
    print(line, flush=True)


def main() -> int:
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm
    from reference import pre_process_fwd_kernel_merged as ref_ppfm

    dev = "npu:0"
    K = V = 128
    BT = 64
    cfgs = [
        (256, 1, 1, [0, 256], "HV=1"),
        (256, 2, 2, [0, 256], "HV=2"),
        (256, 4, 4, [0, 256], "HV=4"),
        (256, 8, 8, [0, 256], "HV=8"),
        (512, 2, 2, [0, 512], "HV=2 T=512"),
        (128, 2, 2, [0, 128], "HV=2 2chunk"),
        (192, 2, 2, [0, 192], "HV=2 3chunk"),
    ]
    for T, HK, HV, cu, tag in cfgs:
        for rep in range(2):
            one(dev, ref_ppfm, npu_ppfm, T, HK, HV, K, V, BT, cu, tag=f"{tag}#{rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
