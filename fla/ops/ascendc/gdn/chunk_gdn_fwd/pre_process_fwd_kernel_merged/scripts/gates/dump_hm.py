"""L1 位级回归：跑固定用例集，把 hm 存成 .pt（同 seed / 同 tiling）。

用法：ASCEND_RT_VISIBLE_DEVICES=7 python run_npu.py dump_hm.py <op_dir> <tag>
产物：<op_dir>/scripts/.cache/hm/<tag>/<case>.pt
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

CASES = [
    ("gdn-t256", 256, 2, 2, "gdn", [0, 256]),
    ("kda-t256", 256, 2, 2, "kda", [0, 256]),
    ("gdn-t1023", 1023, 2, 2, "gdn", [0, 1023]),
    ("gdn-hv8", 256, 2, 8, "gdn", [0, 256]),
    ("gdn-sub", 512, 2, 2, "gdn", [40, 512]),
]


def load_smoke(op_dir: str):
    path = os.path.join(op_dir, "scripts", "npu_smoke_ppfm.py")
    spec = importlib.util.spec_from_file_location("ppfm_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ppfm_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    op_dir, tag = sys.argv[1], sys.argv[2]
    dev = "npu:0"
    out = os.path.join(op_dir, "scripts", ".cache", "hm", tag)
    os.makedirs(out, exist_ok=True)
    smoke = load_smoke(op_dir)
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    for i, (name, T, HK, HV, variant, cu) in enumerate(CASES):
        case = smoke.build_case(2000 + i, T, HK, HV, 128, 128, variant, BT=64)
        npu_in = {k: (smoke.to_npu_bnsd(v) if v is not None else None)
                  for k, v in case.items()}
        kwargs = dict(cu_seqlens=list(cu), chunk_size=64)
        if variant == "gdn":
            kwargs["g"] = npu_in["g"].to(dev)
            kwargs["gk"] = None
        else:
            kwargs["gk"] = npu_in["gk"].to(dev)
            kwargs["g"] = None
        got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev),
                       npu_in["u"].to(dev), **kwargs)
        torch.npu.synchronize()
        torch.save(got.cpu(), os.path.join(out, name + ".pt"))
        print(f"saved {name} shape={tuple(got.shape)}")
    print("DUMP_DONE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
