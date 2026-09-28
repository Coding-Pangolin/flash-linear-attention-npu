"""复现 smoke 里 gdn-single 的失败：逐步去掉差异项，定位触发条件。

    python3 npu_repro.py --variants ref-first,noref-first --reps 3
"""

from __future__ import annotations

import argparse
import sys

import torch

OP = ("/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
      "chunk_gdn_fwd/pre_process_fwd_kernel_merged")
sys.path.insert(0, OP + "/reference")
sys.path.insert(0, "/tmp")

from npu_one_case import build_case, compare, to_npu_bnsd  # noqa: E402


def run(ref_ppfm, npu_ppfm, dev, cpu_first: bool, alloc_pad: int, tag: str, gfirst=False):
    T, K, V, BT, HK, HV, cu = 256, 128, 128, 64, 2, 2, [0, 256]
    case = build_case(1000, T, HK, HV, K, V, "gdn", BT=BT)
    if alloc_pad:
        pad = torch.zeros(alloc_pad, dtype=torch.float32, device=dev)
    want = None
    if cpu_first:
        want = torch.stack([
            ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=None,
                     chunk_size=BT, cu_seqlens=list(cu))])
    npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in case.items()}
    if gfirst:
        kwargs = dict(cu_seqlens=list(cu), chunk_size=BT)
        kwargs["g"] = npu_in["g"].to(dev)
        kwargs["gk"] = None
        got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev), **kwargs)
    else:
        got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
                       cu_seqlens=list(cu), chunk_size=BT, g=npu_in["g"].to(dev), gk=None)
    ptrs = " ".join(f"{n}={int(npu_in[n].to(dev).data_ptr()) & 0xfffff:x}"
                    for n in ("k", "w", "u", "g"))
    hmptr = int(got.data_ptr()) & 0xfffff
    torch.npu.synchronize()
    got = got.cpu()
    if want is None:
        want = torch.stack([
            ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=None,
                     chunk_size=BT, cu_seqlens=list(cu))])
    if alloc_pad:
        del pad
    line = f"{tag:28s} hm={hmptr:06x} {ptrs}"
    for hv in range(HV):
        sh = compare(got[0, hv, :, :V], want[0, hv, :, :V])
        sm = compare(got[0, hv, :, V:], want[0, hv, :, V:])
        line += f" [h{hv}] h:{sh['matched']:.3f}/{sh['max_abs']:.1e}" \
                f" m:{sm['matched']:.3f}/{sm['max_abs']:.1e}"
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm
    from reference import pre_process_fwd_kernel_merged as ref_ppfm

    dev = "npu:0"
    for rep in range(args.reps):
        run(ref_ppfm, npu_ppfm, dev, False, 0, f"npu-first#{rep}")
        run(ref_ppfm, npu_ppfm, dev, False, 0, f"npu-first-gfirst#{rep}", gfirst=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
