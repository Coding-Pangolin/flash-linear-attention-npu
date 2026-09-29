"""L3 竞态探针：同一用例复跑 N 次（独立进程内多次调用），每次与 CPU 标杆比，
记录 h/m 两半的 matched / max_abs 分布。

用法：
  ASCEND_RT_VISIBLE_DEVICES=7 python run_npu.py race_probe.py <op_dir> [rounds] \
      [T] [HK] [HV] [variant] [seed] [cu0] [cu1]
默认：T=1023 HK=2 HV=2 gdn seed=2008 cu=[0,1023]（实测能复现 GDN h 竞态）
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401  （必须在 .to("npu") 之前导入，否则新容器上 'npu' 未注册）


def load_smoke(op_dir: str):
    path = os.path.join(op_dir, "scripts", "npu_smoke_ppfm.py")
    spec = importlib.util.spec_from_file_location("ppfm_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ppfm_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    op_dir = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 1023
    HK = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    HV = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    variant = sys.argv[6] if len(sys.argv) > 6 else "gdn"
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 2008
    cu = [int(sys.argv[8]), int(sys.argv[9])] if len(sys.argv) > 9 else [0, T]
    dev = "npu:0"

    smoke = load_smoke(op_dir)
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    case = smoke.build_case(seed, T, HK, HV, 128, 128, variant)
    want = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                    chunk_size=64, cu_seqlens=list(cu)).float()
    npu_in = {k: (smoke.to_npu_bnsd(v) if v is not None else None)
              for k, v in case.items()}
    kwargs = dict(cu_seqlens=list(cu), chunk_size=64)
    if variant == "gdn":
        kwargs["g"] = npu_in["g"].to(dev)
        kwargs["gk"] = None
    else:
        kwargs["gk"] = npu_in["gk"].to(dev)
        kwargs["g"] = None
    k, w, u = (npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev))

    print(f"# race_probe T={T} HK={HK} HV={HV} {variant} seed={seed} cu={cu} rounds={rounds}")
    h_bad = m_bad = 0
    for i in range(1, rounds + 1):
        got = npu_ppfm(k, w, u, **kwargs)
        torch.npu.synchronize()
        got = got.float().cpu()
        w4 = want.reshape(got.shape)
        sh = smoke.compare(got[..., :128], w4[..., :128], 1.5e-2, 2e-3)
        sm = smoke.compare(got[..., 128:], w4[..., 128:], 1.5e-2, 2e-3)
        fh = not (sh["matched"] >= 0.999 and sh["max_abs"] <= 0.05)
        fm = not (sm["matched"] >= 0.999 and sm["max_abs"] <= 0.05)
        h_bad += 1 if fh else 0
        m_bad += 1 if fm else 0
        print(f"  round {i:2d}: h matched={sh['matched']:.6f} max_abs={sh['max_abs']:.3e}"
              f"{'  <-- FAIL' if fh else ''} | m matched={sm['matched']:.6f}"
              f" max_abs={sm['max_abs']:.3e}{'  <-- FAIL' if fm else ''}")
    print(f"# 汇总: h FAIL {h_bad}/{rounds}, m FAIL {m_bad}/{rounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
