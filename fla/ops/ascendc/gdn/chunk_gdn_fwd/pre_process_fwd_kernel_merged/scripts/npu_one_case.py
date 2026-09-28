"""单用例上板对拍：定位 multi-seg / 尾块 partial chunk 的问题。

用法：
    python3 npu_one_case.py --cu 0,88,188,256 --hk 2 --hv 2 --variant gdn
    python3 npu_one_case.py --T 256 --cu 0,256 --no-ref     # 只跑 NPU，不对比
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch


def build_case(seed: int, T: int, HK: int, HV: int, K: int, V: int, variant: str, BT: int = 64):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    def real_gate(shape_tail):
        nblk = -(-T // BT)
        base = -0.013 / BT * (1 + torch.rand(nblk, BT, *shape_tail, generator=g) * 0.5)
        return base.cumsum(1).reshape(nblk * BT, *shape_tail)[:T].contiguous()

    k = torch.nn.functional.normalize(randn((T, HK, K)), dim=-1).to(torch.bfloat16)
    v = randn((T, HV, V)).to(torch.bfloat16)
    beta = torch.rand((T, HV, 1), generator=g, dtype=torch.float32) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).to(torch.bfloat16)
    u = v.clone()
    if variant == "gdn":
        return dict(k=k, v=v, w=w, u=u, g=real_gate((HV,)).to(torch.float32), gk=None)
    # KDA：gk 按 value head（HV），k 按 HK 头；HK < HV（GVA）合法
    return dict(k=k, v=v, w=w, u=u, g=None, gk=real_gate((HV, K)).to(torch.float32))


def to_npu_bnsd(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(1, 0).unsqueeze(0).contiguous()


def compare(got, want, atol=1.5e-2, rtol=2e-3):
    got = got.float()
    want = want.float()
    diff = (got - want).abs()
    ok = diff <= (atol + rtol * want.abs())
    denom = want.abs().clamp_min(1e-6)
    return dict(matched=float(ok.float().mean()), max_abs=float(diff.max()),
                mare=float((diff / denom).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", default=os.environ.get("PPFM_OP_DIR",
                   "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
                   "chunk_gdn_fwd/pre_process_fwd_kernel_merged"))
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--hv", type=int, default=2)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--BT", type=int, default=64)
    ap.add_argument("--variant", default="gdn", choices=["gdn", "kda"])
    ap.add_argument("--cu", default="0,88,188,256")
    ap.add_argument("--seed", type=int, default=1004)
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()

    cu = [int(x) for x in args.cu.split(",") if x != ""]
    T, K, V, BT = args.T, args.K, args.V, args.BT
    dev = args.device

    sys.path.insert(0, os.path.join(args.op_dir, "reference"))
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    case = build_case(args.seed, T, args.hk, args.hv, K, V, args.variant, BT=BT)
    npu_in = {kk: (to_npu_bnsd(vv) if vv is not None else None) for kk, vv in case.items()}
    kwargs = dict(cu_seqlens=list(cu), chunk_size=BT)
    if args.variant == "gdn":
        kwargs["g"] = npu_in["g"].to(dev)
        kwargs["gk"] = None
    else:
        kwargs["gk"] = npu_in["gk"].to(dev)
        kwargs["g"] = None

    print(f"cu={cu} variant={args.variant} HK={args.hk} HV={args.hv} "
          f"T={T} K={K} V={V} BT={BT}")
    got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev), **kwargs)
    torch.npu.synchronize()
    got = got.cpu()
    print("npu done, shape =", tuple(got.shape))
    if args.no_ref:
        return 0

    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    want = []
    for bos, eos in zip(cu, cu[1:]):
        want.append(ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                             chunk_size=BT, cu_seqlens=[bos, eos]))
    want = torch.stack(want, dim=0)
    st = compare(got, want)
    print(f"matched={st['matched']:.6f} max_abs={st['max_abs']:.3e} MARE={st['mare']:.3e}")
    for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
        s = compare(got[..., sl], want[..., sl])
        print(f"  {tag}: matched={s['matched']:.6f} max_abs={s['max_abs']:.3e} MARE={s['mare']:.3e}")
    for i in range(got.shape[0]):
        s = compare(got[i], want[i])
        print(f"  seq{i} [{cu[i]},{cu[i+1]}): matched={s['matched']:.6f} max_abs={s['max_abs']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
