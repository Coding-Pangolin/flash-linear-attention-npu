#!/usr/bin/env python3
"""pre_process_fwd_kernel_merged —— 切分方案成本模型（03 阶段设计输入）

只做算术，不依赖硬件。硬件相关的时间估算使用下方 ASSUMPTIONS 里显式声明的常量。

模型（每个 chunk、每个 value head）：

  h 半边（上游 i_col*BS < V 的 program）
      v_decay = W[BT,K] @ H[K,BSv]         -> BT*K*BSv  MAC
      h      += Kt[K,BT] @ Vnew[BT,BSv]    -> K*BT*BSv  MAC
      读 k[BT,K], w[BT,K], v[BT,BSv]

  m 半边（其余 program）
      kw  = Kt[K,BT] @ W[BT,K]             -> K*BT*K    MAC   <- 与列块无关，每个 m program 重复算
      m   = M[K,K] @ m[K,BSm]              -> K*K*BSm   MAC
      读 k[BT,K], w[BT,K]

  合并后（h 与 m 共享左因子 Kt）：一次 Kt[K,BT] @ [Vnew | W][BT,V+K] 同时出两半。
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------- assumptions
ASSUMPTIONS = {
    "cube_mac_per_cycle_per_aic": 4096,   # 16x16x16 每 cycle
    "cube_freq_ghz": 1.65,
    "aic_count": 28,
    "hbm_bw_gb_s": 1500.0,                # 950PR 典型 1.4~1.6 TB/s，取中值
    "bytes_per_elem": 2,                  # BF16
}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class Case:
    name: str
    hv: int
    hk: int
    k: int
    v: int
    t: int
    bt: int = 64

    @property
    def nt(self) -> int:
        return cdiv(self.t, self.bt)

    @property
    def bs(self) -> int:
        return 32 if self.k <= 64 else 64


def unique_bytes_per_head_chunk(c: Case) -> dict[str, float]:
    b = ASSUMPTIONS["bytes_per_elem"]
    # k 只有 HK 个 head，按 HV 摊分；gk 模式 HK==HV
    return {
        "k": c.bt * c.k * b * (c.hk / c.hv),
        "w": c.bt * c.k * b,
        "v": c.bt * c.v * b,
    }


def scenario_upstream_split(c: Case) -> dict:
    """上游做法：h/m 各自按列块切，每个 program 各读一遍 k、w；M_c 每个 m program 各算一遍。"""
    bs = c.bs
    n_h = cdiv(c.v, bs)
    n_m = cdiv(c.k, bs)
    ub = unique_bytes_per_head_chunk(c)

    mac_h = n_h * (c.bt * c.k * bs + c.k * c.bt * bs)
    mac_m = n_m * (c.k * c.bt * c.k) + (c.k * c.k * c.v)  # M@m 按列分块，总量不重复

    rd = n_h * (ub["k"] + ub["w"]) + n_h * (c.bt * bs * ASSUMPTIONS["bytes_per_elem"])
    rd += n_m * (ub["k"] + ub["w"])

    return {
        "scenario": "upstream_split",
        "programs": c.hv * (n_h + n_m),
        "mac_per_head_chunk": mac_h + mac_m,
        "mac_h": mac_h,
        "mac_m": mac_m,
        "read_bytes_per_head_chunk": rd,
        "unique_read_bytes_per_head_chunk": sum(ub.values()),
        "note": f"h/m 各 {n_h}/{n_m} 列块, BS={bs}",
    }


def scenario_fused_sharedM(c: Case) -> dict:
    """合并：列块仍拆，但 k/w 只读一次、M_c 只算一次（需要跨 program 共享中间量）。"""
    bs = c.bs
    n = cdiv(c.v + c.k, bs)
    ub = unique_bytes_per_head_chunk(c)

    mac_h = c.bt * c.k * c.v + c.k * c.bt * c.v
    mac_m = c.k * c.bt * c.k + c.k * c.k * c.k

    rd = ub["k"] + ub["w"] + 0.0  # v 按列块只读自己那一段，合计等于 uv["v"]
    rd += ub["v"]

    return {
        "scenario": "fused_shared_input_M",
        "programs": c.hv * n,
        "mac_per_head_chunk": mac_h + mac_m,
        "mac_h": mac_h,
        "mac_m": mac_m,
        "read_bytes_per_head_chunk": rd,
        "unique_read_bytes_per_head_chunk": sum(ub.values()),
        "note": f"N=V+K 按 {n} 列块, BS={bs}",
    }


def scenario_fused_n256(c: Case) -> dict:
    """合并：每个 head 一个 program，一次 Kt @ [Vnew|W] 出 N=V+K。"""
    ub = unique_bytes_per_head_chunk(c)
    mac_h = c.bt * c.k * c.v + c.k * c.bt * c.v
    mac_m = c.k * c.bt * c.k + c.k * c.k * c.k
    return {
        "scenario": "fused_single_N256",
        "programs": c.hv,
        "mac_per_head_chunk": mac_h + mac_m,
        "mac_h": mac_h,
        "mac_m": mac_m,
        "read_bytes_per_head_chunk": sum(ub.values()),
        "unique_read_bytes_per_head_chunk": sum(ub.values()),
        "note": "每 head 1 个 program, N=V+K=256",
    }


def summarize(c: Case) -> None:
    rows = [scenario_upstream_split(c), scenario_fused_sharedM(c), scenario_fused_n256(c)]
    heads_chunks = c.hv * c.nt
    print(f"\n=== case {c.name}: HV={c.hv} HK={c.hk} K={c.k} V={c.v} T={c.t} BT={c.bt} "
          f"-> NT={c.nt}, BS={c.bs}, head-chunk={heads_chunks} ===")
    hdr = f"{'scenario':<24}{'prog':>6}{'MAC/hd/ck':>12}{'total MAC(G)':>14}{'read(MB)':>10}{'redund':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        total_mac = r["mac_per_head_chunk"] * heads_chunks
        read_mb = r["read_bytes_per_head_chunk"] * heads_chunks / 1e6
        red = r["read_bytes_per_head_chunk"] / r["unique_read_bytes_per_head_chunk"]
        print(f"{r['scenario']:<24}{r['programs']:>6}{r['mac_per_head_chunk']/1e6:>11.2f}M"
              f"{total_mac/1e9:>14.2f}{read_mb:>10.1f}{red:>7.2f}x")

    print("\n  Cube 下界（假设 28 AIC @1.65GHz, 4096 MAC/cyc/AIC）与 DRAM 下界（假设 1.5TB/s）：")
    for r in rows:
        total_mac = r["mac_per_head_chunk"] * heads_chunks
        read_mb = r["read_bytes_per_head_chunk"] * heads_chunks / 1e6
        cube_us = total_mac / (ASSUMPTIONS["aic_count"] * ASSUMPTIONS["cube_mac_per_cycle_per_aic"] *
                              ASSUMPTIONS["cube_freq_ghz"] * 1e9) * 1e6
        dram_us = read_mb / (ASSUMPTIONS["hbm_bw_gb_s"] * 1e3) * 1e6
        print(f"    {r['scenario']:<24} cube>={cube_us:8.1f} us   dram>={dram_us:8.1f} us"
              f"   serial={c.nt} 步")
    print(f"\n  串行链：每个 program 沿 chunk 走 {c.nt} 步；并行度上限 = programs 数"
          f"（Cube 核只有 {ASSUMPTIONS['aic_count']} 个）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hv", type=int, default=32)
    ap.add_argument("--hk", type=int, default=32)
    ap.add_argument("--k", type=int, default=128, dest="kdim")
    ap.add_argument("--v", type=int, default=128, dest="vdim")
    ap.add_argument("--t", type=int, default=11264)
    ap.add_argument("--bt", type=int, default=64)
    a = ap.parse_args()
    summarize(Case("assumed-model-case", a.hv, a.hk, a.kdim, a.vdim, a.t, a.bt))
    summarize(Case("gk-mode-HK=HV", a.hv, a.hv, a.kdim, a.vdim, a.t, a.bt))
    summarize(Case("g-mode-HK=HV/2", a.hv, max(1, a.hv // 2), a.kdim, a.vdim, a.t, a.bt))


if __name__ == "__main__":
    main()