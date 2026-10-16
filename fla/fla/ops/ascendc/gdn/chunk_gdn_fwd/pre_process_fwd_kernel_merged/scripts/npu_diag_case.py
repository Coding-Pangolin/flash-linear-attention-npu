"""某条泛化用例的失败定位：按 (段, head, 行块, h/m) 拆开看错误分布。

    python3 npu_diag_case.py PPFM-37 [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cases_ppfm import (POLICY, compare, parse_cu, ref_cached_path,  # noqa: E402
                            to_npu_bnsd, build_inputs)

OP_DIR = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    cases = json.load(open(OP_DIR / "cases.json", encoding="utf-8"))
    case = next(c for c in cases if c["用例ID"] == args.case)
    cu = parse_cu(case)
    want = torch.load(ref_cached_path(case), map_location="cpu")["hm"]
    inputs = build_inputs(case)
    dev = args.device
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_fn
    npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in inputs.items()}
    kwargs = dict(cu_seqlens=list(cu), chunk_size=int(case.get("chunk_size", 64)))
    if case["算法路径"] == "GDN":
        kwargs["g"] = npu_in["g"].to(dev)
        kwargs["gk"] = None
    else:
        kwargs["gk"] = npu_in["gk"].to(dev)
        kwargs["g"] = None
    kd = npu_in["k"].to(dev)
    wd = npu_in["w"].to(dev)
    ud = npu_in["u"].to(dev)

    V, K = int(case["V"]), int(case["K"])
    for rep in range(args.repeats):
        got = npu_fn(kd, wd, ud, **kwargs).cpu()
        st = compare(got, want)
        print(f"rep{rep}: matched={st['matched']:.6f} max_abs={st['max_abs']:.3e}")
        if st["matched"] >= 0.9999:
            continue
        # 失败才拆细：先看 h/m，再看段 × 行块
        for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
            s = compare(got[..., sl], want[..., sl])
            print(f"   {tag}: matched={s['matched']:.4f} max_abs={s['max_abs']:.3e}")
        bad = []
        for n in range(got.shape[0]):
            for hv in range(got.shape[1]):
                s = compare(got[n, hv], want[n, hv])
                if s["matched"] < 0.9999:
                    bad.append((n, hv, s["matched"], s["max_abs"]))
        print(f"   坏的 (段,head) 共 {len(bad)}/{got.shape[0] * got.shape[1]}: {bad[:6]}")
        n, hv, _, _ = bad[0]
        for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
            rows = []
            for rb in range(0, K, 16):
                s = compare(got[n, hv, rb:rb + 16, sl], want[n, hv, rb:rb + 16, sl])
                rows.append(f"[{rb:3d}){s['matched']:.2f}/{s['max_abs']:.0e}")
            print(f"   {tag} 行块: " + " ".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
