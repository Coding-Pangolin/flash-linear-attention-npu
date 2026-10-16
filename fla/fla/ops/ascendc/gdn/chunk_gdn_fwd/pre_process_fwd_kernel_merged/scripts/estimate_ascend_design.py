#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pre_process_fwd_kernel_merged —— 昇腾侧设计模型

用 Ascend950PR 的真实平台参数, 回答"h/m 两半该怎么切分"这个设计问题。
H20 的实测值只作为参考打印, 不参与结论。

平台参数来自 docs/precheck.md (读自 Ascend950PR_9579.ini)。

模型口径
--------
* Cube 按 16x16x16 为一拍, 一个 matmul 需要 ceil(M/16)*ceil(N/16)*ceil(Kd/16) 拍
* L0A/L0B 放 matmul 的左右操作数, L0C 放 FP32 累加器, L1 放搬运中转, UB 是 AIV 侧
* FP32 matmul: Ascend Cube 的原生甜点是 BF16/FP16。本模型按"BF16 三段拆分
  (hi/lo, 3 次 bf16 乘)"估算 FP32 乘的代价, 这是保守估计; 若平台有原生 FP32 通道
  则按 1 次计, 用 --fp32-passes 切换
"""

from __future__ import annotations

import argparse


PLATFORM = dict(
    name="Ascend950PR_9579 (DAV-3510)",
    aic=28, aiv=56, freq_ghz=1.65, cube_tile=16,
    l0a_kb=64, l0b_kb=64, l0c_kb=256, l1_kb=512, ub_kb=248,
    l2_mb=128, hbm_gb_s=1500.0,
)

# H20 实测值, 仅作参考打印 (来自 nvtx_gpu_proj_sum, 220 次 launch 的均值)
H20_REF = {
    "model-g default": 1549.9,
    "model-gk default": 1608.8,
    "model-dplr default": 1609.6,
}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def cube_cycles(M: int, N: int, Kd: int) -> int:
    t = PLATFORM["cube_tile"]
    return cdiv(M, t) * cdiv(N, t) * cdiv(Kd, t)


def kb(nbytes: float) -> float:
    return nbytes / 1024.0


class Case:
    def __init__(self, hv, hk, k, v, t, bt):
        self.hv, self.hk, self.k, self.v, self.t, self.bt = hv, hk, k, v, t, bt

    @property
    def nt(self):
        return cdiv(self.t, self.bt)

    @property
    def bs(self):
        return 32 if self.k <= 64 else 64


def step_ops_upstream(c: Case, fp32_passes: int):
    """上游式: h program 覆盖 V 的 BS 列, m program 覆盖 K 的 BS 列。"""
    bs, K, V, BT = c.bs, c.k, c.v, c.bt
    h_ops = [
        dict(name="w@h", M=BT, N=bs, Kd=K, passes=1,
             l0a=BT * K * 2, l0b=K * bs * 2, l0c=BT * bs * 4),
        dict(name="kT@v", M=K, N=bs, Kd=BT, passes=1,
             l0a=K * BT * 2, l0b=BT * bs * 2, l0c=K * bs * 4),
    ]
    m_ops = [
        dict(name="kT@w", M=K, N=K, Kd=BT, passes=1,
             l0a=K * BT * 2, l0b=BT * K * 2, l0c=K * K * 4),
        dict(name="M@m(fp32)", M=K, N=bs, Kd=K, passes=fp32_passes,
             l0a=K * K * 4, l0b=K * bs * 4, l0c=K * bs * 4),
    ]
    return h_ops, m_ops


def step_ops_merged(c: Case, fp32_passes: int):
    """合并式: 一个 program 同时算两半, 串行链变成 3 个 matmul。"""
    K, V, BT = c.k, c.v, c.bt
    return [
        dict(name="w@h", M=BT, N=V, Kd=K, passes=1,
             l0a=BT * K * 2, l0b=K * V * 2, l0c=BT * V * 4),
        dict(name="kT@[v|w]", M=K, N=V + K, Kd=BT, passes=1,
             l0a=K * BT * 2, l0b=BT * (V + K) * 2, l0c=K * (V + K) * 4),
        dict(name="M@m(fp32)", M=K, N=K, Kd=K, passes=fp32_passes,
             l0a=K * K * 4, l0b=K * K * 4, l0c=K * K * 4),
    ]


def report(c: Case, fp32_passes: int):
    p = PLATFORM
    h_ops, m_ops = step_ops_upstream(c, fp32_passes)

    print(f"\n{'=' * 92}")
    print(f"case: HV={c.hv} HK={c.hk} K={c.k} V={c.v} BT={c.bt} T={c.t} "
          f"-> NT={c.nt}  BS={c.bs}  fp32_passes={fp32_passes}")
    print(f"平台: {p['name']}  AIC={p['aic']} AIV={p['aiv']} @{p['freq_ghz']}GHz  "
          f"L0A/L0B={p['l0a_kb']}/{p['l0b_kb']}KB L0C={p['l0c_kb']}KB "
          f"L1={p['l1_kb']}KB UB={p['ub_kb']}KB")
    print("=" * 92)

    def cycles(ops):
        return sum(cube_cycles(o["M"], o["N"], o["Kd"]) * o["passes"] for o in ops)

    cyc_h, cyc_m = cycles(h_ops), cycles(m_ops)
    print("\n[1] 每步 Cube 拍数(16x16 粒度, FP32 乘按拆分倍数计)")
    print(f"    h program: {cyc_h:4d} 拍   " +
          " + ".join(f"{o['name']}={cube_cycles(o['M'], o['N'], o['Kd']) * o['passes']}"
                     for o in h_ops))
    print(f"    m program: {cyc_m:4d} 拍   " +
          " + ".join(f"{o['name']}={cube_cycles(o['M'], o['N'], o['Kd']) * o['passes']}"
                     for o in m_ops))
    n_h, n_m = cdiv(c.v, c.bs), cdiv(c.k, c.bs)
    prog_split = c.hv * (n_h + n_m)
    print(f"    上游分流: 每 head {n_h}+{n_m} 个 program, 共 {prog_split} 个, "
          f"铺在 {p['aic']} 个 AIC 上 = {prog_split / p['aic']:.2f} 波")

    serial_split = c.nt * max(cyc_h, cyc_m)
    total_split = c.hv * c.nt * (n_h * cyc_h + n_m * cyc_m)
    thr_split = total_split / p["aic"]
    print("\n[2] 串行纯计算下界 vs 吞吐下界(只算 Cube 拍, 不含访存/同步/流水气泡)")
    print(f"    上游分流  串行链 {c.nt} x max({cyc_h},{cyc_m}) = {serial_split} 拍 = "
          f"{serial_split / p['freq_ghz'] / 1e3:8.1f} us")
    print(f"              总算力 {total_split / 1e6:8.2f} M 拍 / {p['aic']} AIC = "
          f"{thr_split / 1e6:8.2f} M 拍 = {thr_split / p['freq_ghz'] / 1e3:8.1f} us")
    serial_shared = c.nt * (cyc_h + cyc_m)          # 两半争同一批核, 串行推进
    print(f"              两半串行(共用核) {c.nt} x ({cyc_h}+{cyc_m}) = {serial_shared} 拍 = "
          f"{serial_shared / p['freq_ghz'] / 1e3:8.1f} us")
    which = "受吞吐限制" if thr_split > serial_shared else "受串行链限制"
    print(f"    -> 上游分流在昇腾上{which} (吞吐/串行 = {thr_split / serial_shared:.2f}x)")

    merged = step_ops_merged(c, fp32_passes)
    cyc_merged = cycles(merged)
    serial_merged = c.nt * cyc_merged
    thr_merged = c.hv * c.nt * cyc_merged / p["aic"]
    print(f"    合并式    串行链 {c.nt} x {cyc_merged} = {serial_merged} 拍 = "
          f"{serial_merged / p['freq_ghz'] / 1e3:8.1f} us   " +
          " + ".join(f"{o['name']}={cube_cycles(o['M'], o['N'], o['Kd']) * o['passes']}"
                     for o in merged))
    print(f"              总算力 {thr_merged / p['freq_ghz'] / 1e3:8.1f} us")
    print(f"    -> 合并把串行链变成 {serial_merged / serial_split:.2f}x, "
          f"总算力变成 {thr_merged / thr_split:.2f}x")

    print("\n[3] 片上容量(单 program 每步的峰值需求, 对比 950PR 物理容量)")
    print(f"    {'program':10s} {'L0A':>9s} {'L0B':>9s} {'L0C(累加器合计)':>16s}   判定")
    for tag, ops in (("h", h_ops), ("m", m_ops), ("merged", merged)):
        l0a = max(o["l0a"] for o in ops)
        l0b = max(o["l0b"] for o in ops)
        l0c = sum(o["l0c"] for o in ops)
        bad = []
        if kb(l0a) > p["l0a_kb"]:
            bad.append("L0A")
        if kb(l0b) > p["l0b_kb"]:
            bad.append("L0B")
        if kb(l0c) > p["l0c_kb"]:
            bad.append("L0C")
        verdict = "OK" if not bad else f"超限: {','.join(bad)} -> 必须切分"
        print(f"    {tag:10s} {kb(l0a):8.1f}K {kb(l0b):8.1f}K {kb(l0c):15.1f}K   {verdict}")

    b = 2
    uniq = c.bt * c.k * b * (c.hk / c.hv) + c.bt * c.k * b + c.bt * c.v * b
    uniq_total = uniq * c.hv * c.nt
    rd_split = c.hv * c.nt * ((n_h + n_m) * (c.bt * c.k * b * (c.hk / c.hv) + c.bt * c.k * b)
                              + n_h * c.bt * c.bs * b)
    rd_merged = c.hv * c.nt * uniq
    print("\n[4] 读入流量(bf16 输入)")
    print(f"    唯一数据 {uniq / 1024:7.1f} KB/head/步")
    print(f"    上游分流 {rd_split / 1e6:8.1f} MB = {rd_split / uniq_total:4.2f}x  "
          f"(DRAM 下界 {rd_split / (p['hbm_gb_s'] * 1e9) * 1e6:7.1f} us)")
    print(f"    合并式   {rd_merged / 1e6:8.1f} MB = {rd_merged / uniq_total:4.2f}x  "
          f"(DRAM 下界 {rd_merged / (p['hbm_gb_s'] * 1e9) * 1e6:7.1f} us)")

    print("\n[5] AIC / AIV 配比(950PR 为 1:2)")
    print(f"    h program: Cube {cyc_h} 拍 + Vector(减法/decay 缩放/BF16 转换, "
          f"约 4 x BT x BS 元素)")
    print(f"    m program: Cube {cyc_m} 拍 + Vector(对角构造/取负, 约 K x BS 元素)")
    print(f"    -> m 的 Cube 负载是 h 的 {cyc_m / cyc_h:.1f}x, 但 Vector 负载远轻; "
          f"按列块分流时两类 program 的 AIC/AIV 利用率不均衡")

    print("\n[6] H20 实测(仅参考, 不参与上面的结论)")
    for name, us in H20_REF.items():
        print(f"    {name:18s} {us:8.1f} us   每步 {us / c.nt:5.2f} us")
    print("    H20 峰值 BF16 约 148 TFLOPS(78 SM), 实测有效算力 43~51 TFLOPS = ~30%,")
    print("    且其耗时远高于自身 Cube 下界 -> H20 上是延迟受限。")
    print("    昇腾这边吞吐下界反而高于串行纯计算下界 -> 结论方向不同, 不能照搬。")

    print("\n[7] 半边拆解对照(H20 实测, 见 benchmarks/cp/README_pre_process_h20.md)")
    h20_h = {"default": 388.1, "tf32x3": 392.8, "ieee": 401.7}
    h20_m = {"default": 1058.1, "tf32x3": 2564.4, "ieee": 16465.2}
    h20_b = {"default": 1620.7, "tf32x3": 3404.7, "ieee": 18321.6}
    print(f"    {'precision':<10}{'h(us)':>9}{'m(us)':>10}{'h+m':>10}{'both':>10}"
          f"{'both/max':>10}{'both/sum':>10}")
    for key in h20_h:
        s, mx = h20_h[key] + h20_m[key], max(h20_h[key], h20_m[key])
        print(f"    {key:<10}{h20_h[key]:9.1f}{h20_m[key]:10.1f}{s:10.1f}{h20_b[key]:10.1f}"
              f"{h20_b[key] / mx:10.2f}{h20_b[key] / s:10.2f}")
    print(f"    MAC 比 m/h = 2.00x;  耗时比 m/h: default {h20_m['default']/h20_h['default']:.2f}x, "
          f"ieee {h20_m['ieee']/h20_h['ieee']:.1f}x")
    print("    => both 约等于 h+m, 而不是 max(h,m): 两半在 H20 上并没有并行起来")
    print("       (128 个 program 铺在 78 个 SM 上只有 1.64 波; 昇腾 128/28 AIC = 4.57 波, 更不可能并行)")
    print("    => h 半边对 precision 几乎不敏感(+3.5%), 因为它的两个 dot 输入都是 BF16;")
    print("       m 半边差 41x, 全部来自 M_c @ m 这个 FP32 乘")


def main():
    ap = argparse.ArgumentParser(description="昇腾侧设计模型")
    ap.add_argument("--hv", type=int, default=32)
    ap.add_argument("--hk", type=int, default=32)
    ap.add_argument("--k", type=int, default=128, dest="kdim")
    ap.add_argument("--v", type=int, default=128, dest="vdim")
    ap.add_argument("--t", type=int, default=11264)
    ap.add_argument("--bt", type=int, default=64)
    ap.add_argument("--fp32-passes", type=int, default=3,
                    help="FP32 matmul 的 BF16 拆分次数(3=bf16x3, 1=平台原生 FP32)")
    a = ap.parse_args()
    report(Case(a.hv, a.hk, a.kdim, a.vdim, a.t, a.bt), a.fp32_passes)


if __name__ == "__main__":
    main()
