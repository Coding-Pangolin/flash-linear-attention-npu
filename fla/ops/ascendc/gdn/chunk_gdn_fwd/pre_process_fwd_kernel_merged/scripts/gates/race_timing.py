"""进程级 soak 的成本归因：ACL 初始化 / CPU 标杆 / kernel 各占多少。

2026-09-29 在 241 实测（单进程 ~3.5 min）：
    ACL/torch_npu 初始化 ~150 s、5 用例 CPU 标杆 ~58 s、kernel 5 次实跑 + 对拍 ~1.9 s
⇒ 进程级探针"贵"在冷启动，不在算子上。改探针成本模型前先跑这个。

用法: ASCEND_RT_VISIBLE_DEVICES=1 python race_timing.py <op_dir>
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  （必须在 .to("npu") 之前导入）


def _stamp(t0: float, label: str) -> float:
    now = time.time()
    print(f"[timing] {label:28s} {now - t0:8.2f} s", flush=True)
    return now


def _load_module(path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    op_dir = sys.argv[1]
    t = time.time()
    t = _stamp(t, "import torch/torch_npu")

    smoke = _load_module(os.path.join(op_dir, "scripts", "npu_smoke_ppfm.py"), "ppfm_smoke")
    dump = _load_module(os.path.join(op_dir, "scripts", "gates", "dump_hm.py"), "dump_hm")
    t = _stamp(t, "load smoke+cases 模块")

    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402
    t = _stamp(t, "import fla_npu op")

    prepared = []
    for i, (name, T, HK, HV, variant, cu) in enumerate(dump.CASES):
        tc = time.time()
        case = smoke.build_case(2000 + i, T, HK, HV, 128, 128, variant, BT=64)
        want = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                        chunk_size=64, cu_seqlens=list(cu)).float()
        npu_in = {k: (smoke.to_npu_bnsd(v) if v is not None else None)
                  for k, v in case.items()}
        kwargs = dict(cu_seqlens=list(cu), chunk_size=64)
        if variant == "gdn":
            kwargs["g"] = npu_in["g"].to("npu:0")
            kwargs["gk"] = None
        else:
            kwargs["gk"] = npu_in["gk"].to("npu:0")
            kwargs["g"] = None
        prepared.append((name, npu_in["k"].to("npu:0"), npu_in["w"].to("npu:0"),
                         npu_in["u"].to("npu:0"), kwargs, want))
        print(f"[timing]   prep {name:10s} {time.time() - tc:8.2f} s", flush=True)
    t = _stamp(t, "prep 全 5 例（含 CPU 标杆）")

    for name, k, w, u, kwargs, want in prepared:
        tk = time.time()
        got = npu_ppfm(k, w, u, **kwargs)
        torch.npu.synchronize()
        got = got.float().cpu()
        w4 = want.reshape(got.shape)
        sh = smoke.compare(got[..., :128], w4[..., :128], 1.5e-2, 2e-3)
        print(f"[timing]   run {name:10s} {time.time() - tk:8.2f} s "
              f"h max={sh['max_abs']:.3e}", flush=True)
    _stamp(t, "kernel 5 次实跑 + 对拍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
