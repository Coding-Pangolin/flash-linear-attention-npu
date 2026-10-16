#!/usr/bin/env python3
"""性能基线保存与 A/B 对比（配套 docs/refactor_iteration_protocol.md §3）。

为什么要这个脚本：`msprof op` 是**单次发射**，本身有噪声，而且与稳态 device event 差约 1.27×。
所以"没退化"必须有量化判据：**同卡同频 + 每 shape 重复 N 次取中位 + 阈值与极差双条件**。

用法（容器内，先 source set_env.sh）：
    # 1) 留基线（把当前已装 TAG 的 OPP 采一遍）
    python3 perf_ab.py --save --name refactor_base --repeat 3
    # 2) 待测批次：先 exp_switch.sh 切到 TAG，再对比
    python3 perf_ab.py --compare --name refactor_base --tag t3c_a --tol 0.02
    # KDA 单独一组基线（每 chunk 成本与 GDN 差 ~2×）
    python3 perf_ab.py --save --name refactor_base --variant kda

产物：<op_dir>/scripts/.cache/perf/<name>[_<variant>].json
判据：REGRESS 需**同时**满足 —— median 超 (1+tol)，且 差值 > 两侧极差（避免把噪声当回退）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_SHAPES = ["1024:8", "4096:8", "11264:32"]
KERNEL_NAME = "PreProcessFwdKernelMerged"
HERE = Path(__file__).resolve().parent


def parse_opbasic(csv_path: Path) -> tuple[float, int | None, int]:
    """返回 (Task Duration 中位/取最大, Current Freq, 命中的行数)。"""
    durs: list[float] = []
    freqs: list[int] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        card = csv.reader(fh)
        header = None
        for row in card:
            if not row or all(not c.strip() for c in row):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = dict(zip(header, row))
            name = rec.get("Op Name", "")
            if KERNEL_NAME not in name:
                continue
            try:
                durs.append(float(rec.get("Task Duration(us)", "nan")))
                freqs.append(int(float(rec.get("Current Freq", "0") or 0)))
            except ValueError:
                continue
    if not durs:
        raise RuntimeError(f"{csv_path} 里没有 {KERNEL_NAME} 的行（多核会有多行，本题取最大）")
    # 同一 kernel 可能有多行（不同 core type）；取最大即主 kernel 时长
    return max(durs), (freqs[0] if freqs else None), len(durs)


def run_once(op_dir: Path, shape: str, variant: str, dev: str, keep: Path | None) -> tuple[float, int | None]:
    t_len, hv = shape.split(":")
    out_dir = Path(tempfile.mkdtemp(prefix="perfab_"))
    env = dict(os.environ)
    env.update({
        "PPFM_T": t_len, "PPFM_HK": hv, "PPFM_HV": hv, "PPFM_VARIANT": variant,
        "PPFM_DEV": dev,
    })
    cmd = [
        "msprof", "op",
        f"--application={sys.executable} {HERE / 'perf_harness.py'}",
        f"--output={out_dir}", f"--kernel-name={KERNEL_NAME}",
        "--launch-count=1", "--warm-up=1", "--aic-metrics=Default",
    ]
    subprocess.run(cmd, env=env, cwd=str(op_dir), stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT, check=False)
    hits = list(out_dir.rglob("OpBasicInfo.csv"))
    if not hits:
        raise RuntimeError(f"msprof 没产出 OpBasicInfo.csv（shape={shape} variant={variant}）")
    dur, freq, rows = parse_opbasic(hits[0])
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copytree(hits[0].parent, keep / f"{shape.replace(':', 'x')}_{int(time.time())}",
                        dirs_exist_ok=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    if rows > 1:
        print(f"  [note] {shape} 命中 {rows} 行（多 core type），取最大 {dur:.1f} us", file=sys.stderr)
    return dur, freq


def measure(op_dir: Path, shapes: list[str], variant: str, repeat: int,
            dev: str, keep: Path | None) -> dict:
    result: dict = {"variant": variant, "repeat": repeat, "shapes": {}}
    for shape in shapes:
        runs, freqs = [], []
        for _ in range(repeat):
            dur, freq = run_once(op_dir, shape, variant, dev, keep)
            runs.append(dur)
            if freq:
                freqs.append(freq)
            print(f"  [perf] {shape:>12s} run={dur:9.2f} us", flush=True)
        result["shapes"][shape] = {
            "median": statistics.median(runs),
            "min": min(runs), "max": max(runs),
            "spread": max(runs) - min(runs),
            "runs": runs,
            "freq": freqs[0] if freqs else None,
        }
    return result


def installed_kernel_md5(op_dir: Path) -> str:
    import hashlib
    for p in sorted(Path(op_dir).rglob("*.o")):
        if KERNEL_NAME in p.name or "pre_process" in p.name.lower():
            return hashlib.md5(p.read_bytes()).hexdigest()[:16]
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", default=str(HERE.parent.parent))
    ap.add_argument("--name", required=True, help="基线名（save/compare 同一个名字）")
    ap.add_argument("--tag", default="", help="本次待测 TAG（只写进报告，便于追溯）")
    ap.add_argument("--variant", default="gdn", choices=["gdn", "kda"])
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.02, help="允许的最大回退比例，默认 2%%")
    ap.add_argument("--dev", default=os.environ.get("PPFM_DEV", "npu:0"))
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--keep-raw", action="store_true", help="保留原始 msprof 目录便于复核")
    args = ap.parse_args()
    if not (args.save ^ args.compare):
        ap.error("必须且只能指定 --save 或 --compare")

    op_dir = Path(args.op_dir).resolve()
    shapes = [s for s in args.shapes.split(",") if s]
    cache = op_dir / "scripts" / ".cache" / "perf"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.variant == "gdn" else f"_{args.variant}"
    base_path = cache / f"{args.name}{suffix}.json"
    keep = (cache / f"raw_{args.tag or args.name}{suffix}") if args.keep_raw else None

    md5 = installed_kernel_md5(op_dir)
    print(f"[perf_ab] op={op_dir}\n          installed kernel .o md5={md5} "
          f"variant={args.variant} repeat={args.repeat} dev={args.dev}")

    cur = measure(op_dir, shapes, args.variant, args.repeat, args.dev, keep)
    cur["tag"] = args.tag or args.name
    cur["kernel_md5"] = md5
    cur["time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.save:
        base_path.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
        print(f"[perf_ab] 基线已保存 -> {base_path}")
        return 0

    if not base_path.exists():
        print(f"[perf_ab] 找不到基线 {base_path}；先 --save", file=sys.stderr)
        return 2
    base = json.loads(base_path.read_text())
    if base.get("kernel_md5") == cur["kernel_md5"]:
        print("[perf_ab] 注意：待测与基线的 kernel .o md5 相同 ⇒ 机器码恒等，"
              "性能必然一致（本可跳过本次采集）")

    print(f"\n{'shape':>12s} {'base(us)':>10s} {'new(us)':>10s} {'delta':>9s} "
          f"{'spread':>16s} {'freq':>10s}  verdict")
    regress = 0
    for shape in shapes:
        b = base["shapes"].get(shape)
        n = cur["shapes"][shape]
        if b is None:
            print(f"{shape:>12s} {'-':>10s} {n['median']:10.2f} {'(no base)':>9s}")
            continue
        delta = (n["median"] - b["median"]) / b["median"] if b["median"] else 0.0
        guard = delta > args.tol and (n["median"] - b["median"]) > max(n["spread"], b["spread"], 1e-9)
        verdict = "REGRESS" if guard else ("faster" if delta < -args.tol else "PASS")
        regress += 1 if guard else 0
        print(f"{shape:>12s} {b['median']:10.2f} {n['median']:10.2f} {delta:+9.2%} "
              f"{b['spread']:7.2f}/{n['spread']:<7.2f} "
              f"{int(b.get('freq') or 0):>4d}/{int(n.get('freq') or 0):<4d}  {verdict}")

    print(f"\n[perf_ab] tol={args.tol:.1%}  REGRESS={regress}  "
          f"({'PERF_PASS' if regress == 0 else 'PERF_REGRESS'})")
    return 1 if regress else 0


if __name__ == "__main__":
    raise SystemExit(main())
