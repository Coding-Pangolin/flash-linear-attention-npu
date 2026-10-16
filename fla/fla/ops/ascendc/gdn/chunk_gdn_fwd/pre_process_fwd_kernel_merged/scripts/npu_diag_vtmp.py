"""读 PPFM_DIAG 版本内核写进 hm 的诊断指纹，判断 KDA 概率性 h 错的机制。

每个工作项的 hm[n,hv,0,V:V+K] 的 m 半边第 0 行被诊断覆盖：
  lane 0..3  = AIV 读到的 vTmpF_[0]（第 c 个 chunk）
  lane 8..11 = AIC 写出的 vTmpF_[0]
chunk 0 时 h 初值 = 0，所以 vTmp 必须**恰好为 0**：
  * AIV 读到非 0，且与 AIC 写出的值一致 → AIC 的 mm1 输入（hBf_）是脏的（跨核可见性）
  * AIV 读到非 0，但与 AIC 写出的值不一致 → AIV 读到了别的代（flag 提前/缓存）
用法: python3 npu_diag_vtmp.py PPFM-06 [--heads 0,5,11]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_cases_ppfm import build_inputs, parse_cu, to_npu_bnsd  # noqa: E402

OP_DIR = HERE.parent
DIAG = 4


def run_case(case, ref_fn, npu_fn, dev, repeats, tag=""):
    cu = parse_cu(case)
    V, K = int(case["V"]), int(case["K"])
    inputs = build_inputs(case)
    npu_in = {k: (to_npu_bnsd(v) if v is not None else None) for k, v in inputs.items()}
    kwargs = dict(cu_seqlens=list(cu), chunk_size=int(case.get("chunk_size", 64)))
    if case["算法路径"] == "GDN":
        kwargs["g"] = npu_in["g"].to(dev); kwargs["gk"] = None
    else:
        kwargs["gk"] = npu_in["gk"].to(dev); kwargs["g"] = None
    kd, wd, ud = npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev)

    want = None
    ref_path = HERE / ".cache" / "ref" / f"{case['用例ID']}_v1.pt"
    if ref_path.exists():
        want = torch.load(ref_path, map_location="cpu")["hm"]

    for rep in range(repeats):
        hm = npu_fn(kd, wd, ud, **kwargs).cpu()
        diag = hm[..., 0, V:V + 16].reshape(-1, 16)          # [task, 16]
        aiv = diag[:, :DIAG]
        aic = diag[:, 8:8 + DIAG]
        bad_vtmp0 = (aiv[:, 0].abs() > 1e-6).nonzero().flatten().tolist()
        bad_h = []
        if want is not None:
            got_h = hm[..., :V]
            want_h = want[..., :V]
            for t in range(got_h.shape[0]):
                for hv in range(got_h.shape[1]):
                    d = (got_h[t, hv] - want_h[t, hv]).abs().max().item()
                    if d > 1e-3:
                        bad_h.append((t, hv, round(d, 3)))
        print(f"{tag}rep{rep}: vTmp0≠0 的 task {len(bad_vtmp0)}/{len(aiv)} {bad_vtmp0[:8]}"
              f" | h 错的 (task,head) {len(bad_h)}: {bad_h[:6]}")
        for t in bad_vtmp0[:4]:
            same = bool(torch.allclose(aiv[t], aic[t], atol=1e-6, rtol=1e-3))
            print(f"    task{t}: AIV读到={[f'{x:.3e}' for x in aiv[t].tolist()]} "
                  f"AIC写出={[f'{x:.3e}' for x in aic[t].tolist()]} 一致={same}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="")
    ap.add_argument("--all-kda", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    cases = json.load(open(OP_DIR / "cases.json", encoding="utf-8"))
    dev = args.device
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_fn
    if args.all_kda:
        for case in cases:
            if case["算法路径"] != "KDA":
                continue
            print(f"===== {case['用例ID']} KDA Nseq={case['Nseq']} HK={case['HK']} "
                  f"HV={case['HV']} T={parse_cu(case)[-1]}", flush=True)
            run_case(case, None, npu_fn, dev, args.repeats)
    else:
        case = next(c for c in cases if c["用例ID"] == args.case)
        print(f"===== {args.case}", flush=True)
        run_case(case, None, npu_fn, dev, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
