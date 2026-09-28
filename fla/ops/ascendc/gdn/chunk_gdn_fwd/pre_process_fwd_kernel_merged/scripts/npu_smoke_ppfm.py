"""pre_process_fwd_kernel_merged 上板冒烟 + 精度对拍脚本。

在装有 torch/torch_npu 且能访问 NPU 的容器里运行：

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 npu_smoke_ppfm.py [--op-dir <仓内算子目录>]

对拍方式：同一份随机输入，一路喂仓内 CPU 标杆 reference.py（token-major
``[T,H,K]``），一路 permute 成 ``[1,H,T,D]`` 喂 NPU ctypes 接口；比较 ``hm``。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch


def build_case(seed: int, T: int, HK: int, HV: int, K: int, V: int, variant: str,
               device: str = "cpu", BT: int = 64):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    def real_gate(shape_tail):
        """真实分布：每个 chunk 内做 cumsum 的负对数衰减（与标杆 self_test 同款）。"""
        nblk = -(-T // BT)
        base = -0.013 / BT * (1 + torch.rand(nblk, BT, *shape_tail, generator=g) * 0.5)
        return base.cumsum(1).reshape(nblk * BT, *shape_tail)[:T].contiguous()

    # 模型同构数据（与标杆 self_test 一致）：k 归一化、w = beta * k（beta ~ U(0, 0.02)），
    # 这样 |Kw| 远小于 1、m 链良态。若用满幅随机 w，m = Π M_c 会把 fp32 求和顺序的
    # 1 ulp 差异放大到 O(1)（|m| ~ 1e7），那是用例病态、不是实现缺陷。
    k = torch.nn.functional.normalize(randn((T, HK, K)), dim=-1).to(torch.bfloat16)
    v = randn((T, HV, V)).to(torch.bfloat16)
    beta = torch.rand((T, HV, 1), generator=g, dtype=torch.float32) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).to(torch.bfloat16)
    u = v.clone()
    if variant == "gdn":
        gate = real_gate((HV,)).to(torch.float32)
        return dict(k=k, v=v, w=w, u=u, g=gate, gk=None)
    # KDA：gk 逐 K 门控，按 value head 给（HV 个门控）；k 按 HK 头。HK < HV（GVA）合法。
    gate = real_gate((HV, K)).to(torch.float32)
    return dict(k=k, v=v, w=w, u=u, g=None, gk=gate)


def to_npu_bnsd(x: torch.Tensor) -> torch.Tensor:
    """[T,H] / [T,H,D] -> [1,H,T] / [1,H,T,D]"""
    return x.movedim(1, 0).unsqueeze(0).contiguous()


def compare(got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float):
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    diff = (got - want).abs()
    limit = atol + rtol * want.abs()
    matched = (diff <= limit).float().mean().item()
    # MARE：相对误差均值（用于观察整体量级）
    denom = want.abs().clamp_min(1e-6)
    mare = (diff / denom).mean().item()
    return dict(matched=matched, max_abs=diff.max().item(), mare=mare)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", default=os.environ.get(
        "PPFM_OP_DIR",
        "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
        "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged"))
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    T, K, V, BT = 256, 128, 128, 64
    plan = [
        # (名字, variant, HK, HV, cu_seqlens, 每段单独对拍)
        ("gdn-single", "gdn", 2, 2, [0, T], True, T),
        ("kda-single", "kda", 2, 2, [0, T], True, T),
        ("gva-1x2", "gdn", 1, 2, [0, T], True, T),
        ("gva-2x4-sub", "gdn", 2, 4, [64, 192], True, T),
        ("gdn-multiseg", "gdn", 2, 2, [0, 88, 188, 256], False, T),
        ("kda-multiseg", "kda", 2, 2, [0, 64, 128, 256], False, T),
        # GATE_L2：补便宜但关键的形状（T=1 单 chunk / T=1023 尾块 63 / HV=8 多 block）
        ("t68-tail", "gdn", 2, 2, [0, 68], True, 68),
        ("t1023-tail", "gdn", 2, 2, [0, 1023], True, 1023),
        ("gdn-hv8", "gdn", 2, 8, [0, T], True, T),
        ("kda-hv8", "kda", 2, 8, [0, T], True, T),
    ]

    dev = args.device
    failures = 0
    print("=" * 78)
    print(f"pre_process_fwd_kernel_merged 上板冒烟（device={dev}, T={T}, K={K}, V={V}, BT={BT}）")
    print("=" * 78)
    for idx, (name, variant, hk, hv, cu, single_seg, t_case) in enumerate(plan):
        try:
            case = build_case(1000 + idx, t_case, hk, hv, K, V, variant, BT=BT)
        except AssertionError as exc:
            print(f"[SKIP] {name}: {exc}")
            continue

        # ---- CPU 标杆（每段一次，与竞品调用粒度一致）----
        want = []
        for bos, eos in zip(cu, cu[1:]):
            aw = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                          chunk_size=BT, cu_seqlens=[bos, eos])
            want.append(aw)
        want = torch.stack(want, dim=0)  # [Nseq, HV, K, V+K]

        # ---- NPU ----
        npu_in = {key: (to_npu_bnsd(val) if val is not None else None)
                  for key, val in case.items()}
        kwargs = dict(cu_seqlens=list(cu), chunk_size=BT)
        if variant == "gdn":
            kwargs["g"] = npu_in["g"].to(dev)
            kwargs["gk"] = None
        else:
            kwargs["gk"] = npu_in["gk"].to(dev)
            kwargs["g"] = None
        got = npu_ppfm(
            npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
            **kwargs,
        )
        torch.npu.synchronize()
        got = got.cpu()

        shape_ok = tuple(got.shape) == (len(cu) - 1, hv, K, V + K)
        stats = compare(got, want, atol=1.5e-2, rtol=2e-3)
        # GATE_L2：matced 与 max_abs 都要过（只卡 matched 会漏掉"少数元素错"的假绿）
        half_ok = True
        for _sl in (slice(0, V), slice(V, V + K)):
            _s = compare(got[..., _sl], want[..., _sl], atol=1.5e-2, rtol=2e-3)
            half_ok = half_ok and _s["matched"] >= 0.999 and _s["max_abs"] <= 0.05
        ok = (shape_ok and stats["matched"] >= 0.999
              and stats["max_abs"] <= 0.05 and half_ok)
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:14s} variant={variant} HK={hk} HV={hv} "
              f"cu={cu} got={tuple(got.shape)} want={tuple(want.shape)}")
        print(f"        matched={stats['matched']:.6f}  max_abs={stats['max_abs']:.3e}  "
              f"MARE={stats['mare']:.3e}")
        # 分半观察
        for tag, sl in (("h_half", slice(0, V)), ("m_half", slice(V, V + K))):
            s = compare(got[..., sl], want[..., sl], atol=1.5e-2, rtol=2e-3)
            print(f"        {tag:6s} matched={s['matched']:.6f} max_abs={s['max_abs']:.3e} "
                  f"MARE={s['mare']:.3e}")

    print("=" * 78)
    print("全部通过" if failures == 0 else f"{failures} 条失败")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
