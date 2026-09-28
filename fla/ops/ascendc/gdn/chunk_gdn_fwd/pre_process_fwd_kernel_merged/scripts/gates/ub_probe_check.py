"""读 PPFM_VTMP_UB_DIAG 探针：跑一个 T=128/HV=1 的 gdn 用例，打印 hm[0,0,0,V:V+8]。

要求 kernel 用 -DPPFM_VTMP_UB=1 -DPPFM_VTMP_UB_DIAG=1 编译（探针只在 chunk 1 采）。
判读：
  idx0 = GM(vTmpF_)[row0], idx1 = GM[row32]
  idx2 = extBlkF_[row0], idx3 = extBlkF_[row32]      （= UB_EXT_F + subIdx*16KB）
  idx4 = vTmpUb_[row0], idx5 = vTmpUb_[row32]        （= UB_EXT_F，共享基址）
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


def main() -> int:
    op_dir = sys.argv[1]
    dev = "npu:0"
    smoke = load_smoke(op_dir)
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    T, HK, HV = 128, 1, 1
    case = smoke.build_case(7, T, HK, HV, 128, 128, "gdn")
    npu_in = {k: (smoke.to_npu_bnsd(v) if v is not None else None)
              for k, v in case.items()}
    got = npu_ppfm(npu_in["k"].to(dev), npu_in["w"].to(dev), npu_in["u"].to(dev),
                   g=npu_in["g"].to(dev), gk=None, cu_seqlens=[0, T], chunk_size=64)
    torch.npu.synchronize()
    got = got.float().cpu()
    row = got[0, 0, 0, 128:136].tolist()
    names = ["GM_row0", "GM_row32", "ownUB_row0", "ownUB_row32",
             "sharedUB_row0", "sharedUB_row32", "unused6", "unused7"]
    for n, v in zip(names, row):
        print(f"  {n:14s} {v:+.6e}")
    print("  诊断判读：shared 两位均与 GM 对应位相等 ⇒ ITER10（共享基址）正确；")
    print("             own 两位与 GM 相等 ⇒ 应改用 per-subcore 偏移（ITER8 的读法）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
