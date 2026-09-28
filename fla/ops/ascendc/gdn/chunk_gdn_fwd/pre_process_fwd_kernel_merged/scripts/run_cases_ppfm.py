"""pre_process_fwd_kernel_merged 泛化用例精度回归（CPU 标杆结果落盘缓存）。

用例表：算子目录 ``cases.json``（41 条：GDN 25 / KDA 16；DPLR 本轮不验收）。
标杆：``reference/reference.py``（唯一可编辑标杆，契约版本见其模块文档）。

用法（在装有 torch/torch_npu 的容器里、算子目录下运行）：

    python3 run_cases_ppfm.py                     # 全量；标杆有缓存就复用
    python3 run_cases_ppfm.py --only PPFM-01,PPFM-11
    python3 run_cases_ppfm.py --ref-only          # 只建标杆缓存（CPU 上跑）
    python3 run_cases_ppfm.py --skip-ref          # 只跑 NPU，必须有缓存
    python3 run_cases_ppfm.py --repeats 5         # 每条用例重复 N 次（查稳定性）

缓存目录：``scripts/.cache/ref/<用例ID>_<数据版本>.pt``，里面存的是
``{"hm": Tensor[Nseq,HV,K,V+K] fp32}`` + 用例指纹；输入按同一种子确定性重放，
所以缓存可以长期复用（改了 ``build_inputs`` 就 bump ``DATA_VERSION``）。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

DATA_VERSION = "v1"
BT_DEFAULT = 64

HERE = Path(__file__).resolve().parent
OP_DIR = HERE.parent
CACHE_DIR = HERE / ".cache" / "ref"

POLICY = {"atol": 1.5e-2, "rtol": 2e-3, "max_abs_limit": 0.05,
          "matched_min": 0.999}


# ---------------------------------------------------------------- 输入构造
def parse_cu(case: dict) -> list[int]:
    """解析用例表的 cu_seqlens。

    多数用例是干净的 JSON（``"[0,1023]"``）；PPFM-34/35 是**带注解的省略写法**
    （``"[0,4096,…,262144]（16 段等长，每段 4096）"``），这里按
    ``Nseq`` + 每段长度重建等长打包序列。
    """
    raw = case["cu_seqlens"].strip()
    try:
        return [int(x) for x in json.loads(raw)]
    except (ValueError, TypeError):
        nseq = int(case["Nseq"])
        m = re.search(r"每段\s*(\d+)", raw)
        seg = int(m.group(1)) if m else int(case["T_win"]) // max(1, nseq)
        return [i * seg for i in range(nseq + 1)]


def build_inputs(case: dict, seed: int = 0):
    """按用例表生成"模型同构"输入（与标杆/GPU 对标同款分布）。

    返回 dict(k, v, u, w, g/gk) —— 全部 CPU、token-major [T, H, D]。
    """
    g = torch.Generator().manual_seed(seed)
    variant = case["算法路径"]
    hk, hv, kdim = int(case["HK"]), int(case["HV"]), int(case["K"])
    vdim = int(case["V"])
    cu = parse_cu(case)
    T = cu[-1]
    bt = int(case.get("chunk_size", BT_DEFAULT))
    ratio = hv // hk
    head_of_k = torch.arange(hv) // ratio

    def randn(shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    k = torch.nn.functional.normalize(randn((T, hk, kdim)), dim=-1).to(torch.bfloat16)
    v = randn((T, hv, vdim)).to(torch.bfloat16)
    beta = torch.rand((T, hv, 1), generator=g, dtype=torch.float32) * 0.02
    w = (beta * k[:, head_of_k].float()).to(torch.bfloat16)

    # gate：每个序列、每个 chunk 内做 cumsum 的负对数衰减（与标杆 self_test 同款）
    def real_gate(tail):
        out = torch.empty((T,) + tail, dtype=torch.float32)
        for bos, eos in zip(cu, cu[1:]):
            nblk = -(-(eos - bos) // bt)
            base = -0.013 / bt * (1 + torch.rand(nblk, bt, *tail, generator=g) * 0.5)
            base = base.cumsum(1).reshape(nblk * bt, *tail)[: eos - bos]
            out[bos:eos] = base
        return out

    dt = torch.bfloat16 if case.get("g/gk dtype") == "BF16" else torch.float32
    if variant == "GDN":
        return dict(k=k, v=v, u=v.clone(), w=w, g=real_gate((hv,)).to(dt), gk=None)
    # KDA（gk）：gk 按 value head 给（[B,T,HV,K]），k 按 HK 头（hk = hv // (HV/HK)）；
    # GVA（HK < HV）合法，与竞品 chunk_kda / CP kernel 一致，不再限制 HK == HV。
    return dict(k=k, v=v, u=v.clone(), w=w, g=None, gk=real_gate((hv, kdim)).to(dt))


def case_fingerprint(case: dict) -> str:
    keys = ("用例ID", "算法路径", "HK", "HV", "K", "V", "cu_seqlens", "chunk_size",
            "g/gk dtype")
    return "|".join(f"{k}={case.get(k)}" for k in keys) + f"|data={DATA_VERSION}"


# ---------------------------------------------------------------- 标杆缓存
def ref_cached_path(case: dict) -> Path:
    return CACHE_DIR / f"{case['用例ID']}_{DATA_VERSION}.pt"


def _ref_worker(args):
    """在子进程里算一个窗口（fork 出来的输入张量按 COW 共享）。"""
    (case, inputs, bos, eos, bt, threads) = args
    if threads:
        torch.set_num_threads(threads)
    from reference import pre_process_fwd_kernel_merged as ref_fn
    bt_eff = min(bt, eos - bos)
    return ref_fn(inputs["k"], inputs["v"], inputs["w"], g=inputs["g"], gk=inputs["gk"],
                  chunk_size=bt_eff, cu_seqlens=[bos, eos])


def _init_worker(threads: int):
    torch.set_num_threads(threads)


def build_reference(case: dict, ref_fn, verbose: bool = True, windows=None):
    """按窗口逐段调用标杆；结果 stack 成 [Nseq, HV, K, V+K]。"""
    inputs = build_inputs(case)
    cu = parse_cu(case)
    bt = int(case.get("chunk_size", BT_DEFAULT))
    t0 = time.perf_counter()
    all_windows = list(zip(cu, cu[1:]))
    sel = all_windows if windows is None else all_windows[windows[0]:windows[1]]
    # 窗口长度 < BT 时标杆要求 chunk_size <= 窗口长度；此时只会有 1 个 chunk，
    # 结果与 chunk_size 无关(语义等价)，因此安全地取 min。
    # ⚠ 不要用 multiprocessing 并行：父进程已经跑过 128 线程的 torch 算子, fork 后
    #   子进程进 torch 会死锁。多段大用例改用"每段一个独立进程 + part 文件"的方案
    #   （见 --windows / --merge-parts）。
    outs = []
    for i, (bos, eos) in enumerate(sel):
        bt_eff = min(bt, eos - bos)
        outs.append(ref_fn(inputs["k"], inputs["v"], inputs["w"], g=inputs["g"],
                           gk=inputs["gk"], chunk_size=bt_eff, cu_seqlens=[bos, eos]))
        if verbose and (i + 1) % 4 == 0:
            print(f"      ... 标杆 {i + 1}/{len(sel)} 段 "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    hm = torch.stack(outs, dim=0)
    if verbose:
        print(f"      标杆完成：{tuple(hm.shape)} {time.perf_counter() - t0:.1f}s",
              flush=True)
    return inputs, hm


def part_path(case: dict, lo: int, hi: int) -> Path:
    return CACHE_DIR / f"{case['用例ID']}_{DATA_VERSION}.w{lo}_{hi}.pt"


def merge_parts(case: dict):
    """把多段标杆的 part 文件按窗口顺序拼成完整缓存；不存在返回 None。"""
    n = len(parse_cu(case)) - 1
    found = {}
    for p in CACHE_DIR.glob(f"{case['用例ID']}_{DATA_VERSION}.w*.pt"):
        lo, hi = p.stem.split(".")[-1][1:].split("_")
        found[(int(lo), int(hi))] = p
    if not found:
        return None
    parts = sorted(found.keys())
    if parts[0][0] != 0 or parts[-1][1] != n:
        return None
    for (a, b), (c, d) in zip(parts, parts[1:]):
        if b != c:
            return None
    hms = [torch.load(found[k], map_location="cpu")["hm"] for k in parts]
    return torch.cat(hms, dim=0)


def load_or_build_reference(case: dict, ref_fn, rebuild: bool = False, windows=None,
                            write_part: bool = False):
    path = ref_cached_path(case)
    if path.exists() and not rebuild:
        blob = torch.load(path, map_location="cpu")
        if blob.get("fingerprint") == case_fingerprint(case):
            return build_inputs(case), blob["hm"], True
    if windows is None and not rebuild:
        merged = merge_parts(case)
        if merged is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"fingerprint": case_fingerprint(case), "hm": merged}, path)
            return build_inputs(case), merged, True
    inputs, hm = build_reference(case, ref_fn, windows=windows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if windows is not None and write_part:
        torch.save({"fingerprint": case_fingerprint(case), "hm": hm},
                   part_path(case, windows[0], windows[1]))
    else:
        torch.save({"fingerprint": case_fingerprint(case), "hm": hm}, path)
    return inputs, hm, False


# ---------------------------------------------------------------- 比对
def compare(got: torch.Tensor, want: torch.Tensor):
    got = got.float()
    want = want.float()
    diff = (got - want).abs()
    limit = POLICY["atol"] + POLICY["rtol"] * want.abs()
    matched = float((diff <= limit).float().mean())
    denom = want.abs().clamp_min(1e-6)
    return {
        "matched": matched,
        "max_abs": float(diff.max()),
        "mare": float((diff / denom).mean()),
    }


def verdict(stats: dict) -> bool:
    return stats["matched"] >= POLICY["matched_min"] and \
        stats["max_abs"] <= POLICY["max_abs_limit"]


def print_breakdown(got: torch.Tensor, want: torch.Tensor, V: int, K: int) -> None:
    """失败用例的定位信息：h/m 两半 + 坏掉的 (段, head) + 行块画像。"""
    for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
        s = compare(got[..., sl], want[..., sl])
        print(f"      {tag}: matched={s['matched']:.4f} max_abs={s['max_abs']:.3e}")
    bad = []
    for n in range(got.shape[0]):
        for hv in range(got.shape[1]):
            s = compare(got[n, hv], want[n, hv])
            if s["matched"] < 0.9999:
                bad.append((n, hv, round(s["matched"], 3), float(f"{s['max_abs']:.2e}")))
    print(f"      坏 (段,head) {len(bad)}/{got.shape[0] * got.shape[1]}: {bad[:8]}")
    if not bad:
        return
    n, hv = bad[0][0], bad[0][1]
    for tag, sl in (("h", slice(0, V)), ("m", slice(V, V + K))):
        rows = []
        for rb in range(0, K, 16):
            s = compare(got[n, hv, rb:rb + 16, sl], want[n, hv, rb:rb + 16, sl])
            rows.append(f"[{rb:3d}){s['matched']:.2f}/{s['max_abs']:.0e}")
        print(f"      {tag} 行块: " + " ".join(rows))


# ---------------------------------------------------------------- 主流程
def to_npu_bnsd(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(1, 0).unsqueeze(0).contiguous()


def main() -> int:
    # ⚠ 标杆是"小算子 + 多线程"的形态：在多进程/高负载机器上必须限流，
    #   否则 128 线程 spin 会互相踩踏（实测把单 case 拖慢 10 倍以上）。
    threads = int(os.environ.get("PPFM_REF_THREADS", "0"))
    if threads <= 0:
        threads = max(1, min(16, (os.cpu_count() or 8) // 8))
    torch.set_num_threads(threads)
    print(f"torch threads = {threads}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OP_DIR / "cases.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--rebuild-ref", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--windows", default="",
                    help="只算窗口区间 A:B（配合 --ref-only 做分段并行标杆）")
    ap.add_argument("--merge-parts", action="store_true",
                    help="只把已有 part 文件合并成完整标杆缓存")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    if only:
        cases = [c for c in cases if c["用例ID"] in only]
    if skip:
        cases = [c for c in cases if c["用例ID"] not in skip]
    cases = [c for c in cases if c["算法路径"] != "DPLR"]

    sys.path.insert(0, str(OP_DIR / "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_fn  # noqa: E402

    print(f"用例 {len(cases)} 条；缓存目录 {CACHE_DIR}")

    win_range = None
    if args.windows:
        lo, hi = (int(x) for x in args.windows.split(":"))
        win_range = (lo, hi)

    if args.merge_parts:
        for case in cases:
            merged = merge_parts(case)
            if merged is None:
                print(f"  {case['用例ID']}: 无完整 part，跳过")
                continue
            torch.save({"fingerprint": case_fingerprint(case), "hm": merged},
                       ref_cached_path(case))
            print(f"  {case['用例ID']}: 合并 {tuple(merged.shape)} -> "
                  f"{ref_cached_path(case).name}")
        return 0

    results = []
    t_start = time.perf_counter()
    for idx, case in enumerate(cases, 1):
        cid = case["用例ID"]
        cu = parse_cu(case)
        tag = (f"{cid} {case['算法路径']} Nseq={len(cu) - 1} HK={case['HK']} "
               f"HV={case['HV']} T={cu[-1]}")
        print(f"[{idx}/{len(cases)}] {tag}", flush=True)
        try:
            if args.skip_ref:
                inputs, want = build_inputs(case), None
                path = ref_cached_path(case)
                if not path.exists():
                    raise RuntimeError(f"缺少标杆缓存 {path}（先跑一次 --ref-only）")
                want = torch.load(path, map_location="cpu")["hm"]
                cached = True
            else:
                inputs, want, cached = load_or_build_reference(
                    case, ref_fn, rebuild=args.rebuild_ref, windows=win_range,
                    write_part=bool(win_range))
            if win_range:
                print(f"      分段标杆 {win_range[0]}:{win_range[1]} 已保存", flush=True)
                results.append(dict(case=cid, ref_only=True, cached=cached))
                continue
            if args.ref_only:
                results.append(dict(case=cid, ref_only=True, cached=cached))
                continue

            from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_fn
            dev = args.device
            npu_in = {kk: (to_npu_bnsd(vv) if vv is not None else None)
                      for kk, vv in inputs.items()}
            kwargs = dict(cu_seqlens=list(cu), chunk_size=int(case.get("chunk_size", BT_DEFAULT)))
            if case["算法路径"] == "GDN":
                kwargs["g"] = npu_in["g"].to(dev)
                kwargs["gk"] = None
            else:
                kwargs["gk"] = npu_in["gk"].to(dev)
                kwargs["g"] = None
            kd = npu_in["k"].to(dev)
            wd = npu_in["w"].to(dev)
            ud = npu_in["u"].to(dev)

            worst = None
            for rep in range(args.repeats):
                got = npu_fn(kd, wd, ud, **kwargs)
                torch.npu.synchronize()
                got = got.cpu()
                stats = compare(got, want)
                shape_ok = tuple(got.shape) == tuple(want.shape)
                ok = shape_ok and verdict(stats)
                if worst is None or stats["matched"] < worst["matched"]:
                    worst = dict(stats, ok=ok, rep=rep, shape_ok=shape_ok)
                if not ok:
                    break
            worst["cached"] = cached
            results.append(dict(case=cid, **worst))
            mark = "PASS" if worst["ok"] else "FAIL"
            print(f"      {mark} matched={worst['matched']:.6f} "
                  f"max_abs={worst['max_abs']:.3e} MARE={worst['mare']:.3e} "
                  f"(rep{worst['rep']}, ref_cached={cached})", flush=True)
            if not worst["ok"] and not worst.get("shape_ok", True):
                print(f"      ⚠ shape 不一致: got={tuple(got.shape)} want={tuple(want.shape)}")
            if not worst["ok"]:
                print_breakdown(got, want, int(case["V"]), int(case["K"]))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append(dict(case=cid, ok=False, error=str(exc)))
            print(f"      ERROR {exc}", flush=True)

    n_fail = sum(1 for r in results if not r.get("ok", True) and not r.get("ref_only"))
    print("=" * 78)
    print(f"合计 {len(results)} 条，失败 {n_fail} 条，用时 {time.perf_counter() - t_start:.0f}s")
    print("=" * 78)
    for r in results:
        if r.get("ref_only"):
            print(f"  {r['case']:10s} ref cached={r['cached']}")
        elif "error" in r:
            print(f"  {r['case']:10s} ERROR {r['error'][:60]}")
        else:
            print(f"  {r['case']:10s} {'PASS' if r['ok'] else 'FAIL'} "
                  f"matched={r['matched']:.6f} max_abs={r['max_abs']:.3e}")
    if args.report:
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print(f"报告 -> {args.report}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
