"""L3 序列探针（**带标杆缓存**）：与 `seq_probe.py` 同序列、同判据，只是 `want` 走盘上缓存。

动机（2026-09-29，241 实测）：单进程成本 ~3.5 min，其中 ACL 初始化 ~150 s、
CPU 标杆 ~58 s、kernel 只 ~1.9 s。标杆缓存后每个进程省 ~58 s，
配合 `race_soak.sh` 的并发批次，16 进程从 ~56 min 压到 ~20 min。

用法:
    python race_probe_cached.py <op_dir> [rounds]
    # 常规入口是 gates/race_soak.sh（它负责 DEV / PROCS / PAR）

缓存位置 `<op_dir>/scripts/.cache/ref_cache/<case>.pt`（`scripts/.cache/` 已在 .gitignore 里）。
改 `reference/reference.py` 或 `dump_hm.CASES` 的 seed/shape 后请删缓存重跑。
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401  （必须在 .to("npu") 之前导入，否则新容器上 'npu' 未注册）


def _load_module(path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    op_dir = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dev = "npu:0"
    cache_dir = os.path.join(op_dir, "scripts", ".cache", "ref_cache")
    os.makedirs(cache_dir, exist_ok=True)

    smoke = _load_module(os.path.join(op_dir, "scripts", "npu_smoke_ppfm.py"), "ppfm_smoke")
    cases = _load_module(os.path.join(op_dir, "scripts", "gates", "dump_hm.py"), "dump_hm").CASES
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm  # noqa: E402
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm  # noqa: E402

    prepared = []
    for i, (name, T, HK, HV, variant, cu) in enumerate(cases):
        case = smoke.build_case(2000 + i, T, HK, HV, 128, 128, variant, BT=64)
        cache_path = os.path.join(cache_dir, f"{name}.pt")
        if os.path.exists(cache_path):
            want = torch.load(cache_path, map_location="cpu")
        else:
            want = ref_ppfm(case["k"], case["v"], case["w"], g=case["g"], gk=case["gk"],
                            chunk_size=64, cu_seqlens=list(cu)).float()
            tmp = f"{cache_path}.{os.getpid()}.tmp"   # 并发批次下多进程同时首跑也安全
            torch.save(want, tmp)
            os.replace(tmp, cache_path)
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
                  f"{'  <-- FAIL' if (fh or fm) else ''}", flush=True)
        total_bad += len(bad)
        print(f"--- round {rd}: FAIL {len(bad)}/{len(prepared)}", flush=True)
    print(f"# 汇总: 总 FAIL {total_bad} / {rounds * len(prepared)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
