"""L3 序列探针：按固定"用例序列"重复跑（复现跨用例顺序相关的竞态），逐例与 CPU 标杆比。

用法: ASCEND_RT_VISIBLE_DEVICES=7 python run_npu.py seq_probe.py <op_dir> [rounds]
序列取 gates/dump_hm.py 的 CASES（实测该序列下 gdn-t1023 曾出现整 head 不同的竞态）。
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


def load_cases(op_dir: str):
    path = os.path.join(op_dir, "scripts", "gates", "dump_hm.py")
    spec = importlib.util.spec_from_file_location("dump_hm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dump_hm"] = mod
    spec.loader.exec_module(mod)
    return mod.CASES


def main() -> int:
    op_dir = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dev = "npu:0"
    smoke = load_smoke(op_dir)
    cases = load_cases(op_dir)
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    prepared = []
    for i, (name, T, HK, HV, variant, cu) in enumerate(cases):
        case = smoke.build_case(2000 + i, T, HK, HV, 128, 128, variant, BT=64)
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
        prepared.append((name, npu_in["k"].to(dev), npu_in["w"].to(dev),
                         npu_in["u"].to(dev), kwargs, want))

    total_bad = 0
    for rd in range(1, rounds + 1):
        bad = []
        for name, k, w, u, kwargs, want in prepared:
            got = npu_ppfm(k, w, u, **kwargs)
            torch.npu.synchronize()
            got = got.float().cpu()
            w4 = want.reshape(got.shape)
            sh = smoke.compare(got[..., :128], w4[..., :128], 1.5e-2, 2e-3)
            sm = smoke.compare(got[..., 128:], w4[..., 128:], 1.5e-2, 2e-3)
            fh = not (sh["matched"] >= 0.999 and sh["max_abs"] <= 0.05)
            fm = not (sm["matched"] >= 0.999 and sm["max_abs"] <= 0.05)
            if fh or fm:
                bad.append(f"{name}: h max={sh['max_abs']:.3e}{' FAIL' if fh else ''}"
                           f" m max={sm['max_abs']:.3e}{' FAIL' if fm else ''}")
            print(f"  rd{rd} {name:10s} h={sh['max_abs']:.3e} m={sm['max_abs']:.3e}"
                  f"{'  <-- FAIL' if (fh or fm) else ''}")
        total_bad += len(bad)
        print(f"--- round {rd}: FAIL {len(bad)}/{len(prepared)}")
    print(f"# 汇总: 总 FAIL {total_bad} / {rounds * len(prepared)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
