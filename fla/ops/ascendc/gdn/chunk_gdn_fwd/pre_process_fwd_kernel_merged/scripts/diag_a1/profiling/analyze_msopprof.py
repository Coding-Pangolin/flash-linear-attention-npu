"""分析 msopprof 上板产物（OpBasicInfo.csv + PipeUtilization.csv + Memory*.csv）。

用法: python3 analyze_msopprof.py <T>:<HV>:<OPPROF目录> [更多...]
  e.g. python3 analyze_msopprof.py 64:1:/data/.../OPPROF_xxx 4096:8:/data/.../OPPROF_yyy

输出的关键量：
  * Task Duration、Block Dim（每 block = 1 AIC + 2 AIV）
  * AIC/AIV 各流水的 busy 时间与占比（占比口径：该子核自己的 elapsed）
  * **估算搬运字节数**：bytes ≈ active_bw(GB/s) × time(us) × 1000
  * 归一化：每 (chunk × head) 的耗时与 GM 写/读字节（用于判断固定开销与可削减的搬运）
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def analyze(tag: str, T: int, HV: int, d: Path) -> None:
    basic = read_csv(d / "OpBasicInfo.csv")[0]
    pu = read_csv(d / "PipeUtilization.csv")
    dur = fnum(basic, "Task Duration(us)")
    nt = -(-T // 64)
    tasks = HV                       # Nseq=1 → Nwork = HV
    per_unit = dur / max(1, nt * tasks)

    print(f"===== {tag}  T={T} HV={HV}  NT={nt} Nwork={tasks}  → {nt * tasks} 个 (chunk×head)")
    print(f"  kernel   : {basic.get('Op Name')}")
    print(f"  duration : {dur:.2f} us   blockDim={basic.get('Block Dim')} "
          f"(mix block dim={basic.get('Mix Block Dim')})  freq={basic.get('Current Freq')}")
    print(f"  归一化   : {per_unit * 1000:.1f} ns / (chunk×head)")

    for row in pu:
        sub = row.get("sub_block_id", "?")
        if sub.startswith("cube"):
            t = fnum(row, "aic_time(us)")
            parts = [(n, fnum(row, f"aic_{n}_time(us)"), fnum(row, f"aic_{n}_ratio"))
                     for n in ("cube", "scalar", "mte1", "mte2", "mte3", "fixpipe")]
            bw = [(n, fnum(row, f"aic_{n}_active_bw(GB/s)")) for n in ("mte1", "mte2", "mte3", "fixpipe")]
        else:
            t = fnum(row, "aiv_time(us)")
            parts = [(n, fnum(row, f"aiv_{n}_time(us)"), fnum(row, f"aiv_{n}_ratio"))
                     for n in ("vec", "scalar", "mte2", "mte3")]
            bw = [(n, fnum(row, f"aiv_{n}_active_bw(GB/s)")) for n in ("mte2", "mte3")]
        det = "  ".join(f"{n}:{(ti or 0):.2f}us/{(r or 0) * 100:.1f}%"
                        for n, ti, r in parts if ti is not None or r is not None)
        print(f"  [{sub:8s}] elapsed={t if t is None else round(t, 2)} us   {det}")
        for n, b in bw:
            if b:
                ti = dict((x[0], x[1]) for x in parts).get(n) or 0.0
                print(f"             {n}: {b:.1f} GB/s → 该流水搬运 ≈ {b * ti * 1000 / 1e6:.3f} MB"
                      f"（每 chunk×head ≈ {b * ti * 1000 / max(1, nt * tasks):.0f} B）")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for spec in sys.argv[1:]:
        T_s, HV_s, path = spec.split(":", 2)
        analyze(Path(path).name, int(T_s), int(HV_s), Path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
