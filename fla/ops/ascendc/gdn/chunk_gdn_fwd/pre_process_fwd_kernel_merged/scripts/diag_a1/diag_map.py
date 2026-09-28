"""逐块/逐点误差映射：定位 tile mmad 输出的误差结构。

用法（容器内）：
    ASCEND_RT_VISIBLE_DEVICES=3 python3 /data/admin123/run_ppfm_npu.py diag_map.py <op_dir>
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch


def load_smoke(op_dir: str):
    path = os.path.join(op_dir, "scripts", "npu_smoke_ppfm.py")
    spec = importlib.util.spec_from_file_location("ppfm_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ppfm_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def block_map(diff: torch.Tensor, block: int = 16, scale: float = 1e-3) -> str:
    rows, cols = diff.shape
    lines = []
    for r0 in range(0, rows, block):
        row = []
        for c0 in range(0, cols, block):
            blk = diff[r0:r0 + block, c0:c0 + block]
            row.append(f"{blk.max().item() / scale:7.1f}")
        lines.append(f"    r{r0:03d}: " + " ".join(row))
    return "\n".join(lines)


def topk_points(got: torch.Tensor, want: torch.Tensor, diff: torch.Tensor, k: int = 5):
    flat = diff.flatten()
    idxs = torch.topk(flat, k).indices.tolist()
    g = got.flatten()
    w = want.flatten()
    out = []
    for idx in idxs:
        r, c = divmod(idx, diff.shape[1])
        rel = abs(g[idx].item() - w[idx].item()) / max(abs(w[idx].item()), 1e-9)
        out.append(f"    ({r:3d},{c:3d}) want={w[idx].item():+.5e} got={g[idx].item():+.5e} "
                   f"diff={flat[idx].item():.2e} rel={rel:.3f}")
    return "\n".join(out)


def main() -> int:
    op_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/admin123/flash-linear-attention-npu/fla/ops/ascendc/gdn/"
        "chunk_gdn_fwd/pre_process_fwd_kernel_merged")
    dev = "npu:0"
    smoke = load_smoke(op_dir)
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    K = V = 128
    BT = 64
    plan = [
        ("T=64-1c-gdn", 64, 1, 1, "gdn"),
        ("T=64-1c-kda", 64, 1, 1, "kda"),
        ("T=128-2c-gdn", 128, 1, 1, "gdn"),
        ("T=256-4c-gdn", 256, 1, 1, "gdn"),
        ("T=256-4c-kda", 256, 1, 1, "kda"),
    ]
    for seed, (name, T, HK, HV, variant) in enumerate(plan, start=7):
        case = smoke.build_case(seed, T, HK, HV, K, V, variant, BT=BT)
        want = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                        chunk_size=BT, cu_seqlens=[0, T])
        npu_in = {k: (smoke.to_npu_bnsd(v) if v is not None else None)
                  for k, v in case.items()}
        kwargs = dict(cu_seqlens=[0, T], chunk_size=BT)
        if variant == "gdn":
            kwargs["g"] = npu_in["g"].to(dev)
            kwargs["gk"] = None
        else:
            kwargs["gk"] = npu_in["gk"].to(dev)
            kwargs["g"] = None
        got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev),
                       npu_in["u"].to(dev), **kwargs)
        torch.npu.synchronize()
        got = got.float().cpu()
        want = want.float().reshape(got.shape)
        print("=" * 78)
        print(f"[{name}] variant={variant} T={T} HK={HK} HV={HV} seed={seed}")
        diff = (got - want).abs()
        tot = smoke.compare(got, want, 1.5e-2, 2e-3)
        print(f"  total matched={tot['matched']:.6f} max_abs={tot['max_abs']:.3e}")
        for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
            g2 = got[0, 0, :, sl]
            w2 = want[0, 0, :, sl]
            d2 = diff[0, 0, :, sl]
            s = smoke.compare(got[..., sl], want[..., sl], 1.5e-2, 2e-3)
            wa = w2.abs()
            offdiag = (wa.sum() - wa.diagonal().sum()).item() / (wa.numel() - wa.shape[0])
            print(f"  [{tag}] matched={s['matched']:.6f} max_abs={s['max_abs']:.3e} "
                  f"|want|max={wa.max().item():.3e} diag_mean={wa.diagonal().mean().item():.3e} "
                  f"offdiag_mean={offdiag:.3e}")
            print(topk_points(g2, w2, d2))
            if tag == "m":
                print("  块图 1e-3：")
                print(block_map(d2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
