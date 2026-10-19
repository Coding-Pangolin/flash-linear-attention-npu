# A2/A3 实测：**AIC 侧完全被 AIV 掩蔽**（2026-10-09，234）

> 机器/口径：`192.168.9.234` / 8×910B3 / device 7 / 1800 MHz，perf case 63
> （B1 HK96 HV96 T8192，chunk 64），`m_a2.sh`（W=2/R=10 与 W=4/R=20 两种口径各测一遍）。
> 复现：`TAG=<tag> CID=63 W=2 R=10 bash m_a2.sh`、`bash point.sh <tag> && CID=63 TAG=<x> bash run_msprof.sh`。

## 1. 为什么要做这一步

`design_a2a3_e2_stage_attribution.md` 用"逐 Stage 单独屏蔽"给出过一份成本分解：

```text
A1 之后 baseline 12,184.7 µs
  V0+V1   4,892 µs  40.1%      C2        1,988 µs  16.3%
  V3      3,075 µs  25.2%      C4+C5+C7  2,246 µs  18.4%
  V6      2,002 µs  16.4%
```

这份表里**所有项相加 = 14,203 µs > 总时长 12,184.7 µs**，本身就说明它不是一份可加的
成本分解（真实数据里不可能有 116%）。A4′（叶子逆外迁 cube）正是基于"**AIC 有 79% 余量、
cube 只占 1.4%**"这条推论，所以本轮先直接测"**把 AIC 的活全摘掉，总时长会掉多少**"。

## 2. 直接测量：屏蔽 AIC 的计算（保留全部核间 flag 回合）

补丁 `patch_probe5.py`：只在 `Process()` 的**调用点**去掉 AIC 的 stage 调用，
**核间 `CrossCoreWait/SetFlag` 一个不动**（C7 的 free 许可是函数体内发的，用等价的
`CrossCoreSetFlag` 补回），所以回合数、flag 配平、数据依赖都保持原样。

| tag | 内容 | event (µs) W2R10 | event (µs) W4R20 |
|---|---|---:|---:|
| `leafpair` | 当前最优（A1 + 叶逆合并） | 11,912.2 | 11,901.3 |
| `probe5a` | **AIC 侧 C2/C4/C5/C7 全部空转** | 12,284.5 | 12,265.6 |
| `probe5c` | **只空转 C2** | 12,328.9 | — |
| `e2v2_base` | 原始基线（无 A1） | — | 18,848.7 |

**结论：把 AIC 的活全部摘掉，总时长不降反升（+370 µs，约 3%）。**
单独摘 C2 也一样（+20 µs，噪声级）。

## 3. msprof 画像对照（同一 case，`--aic-metrics=Default`）

| 指标 (µs) | `leafpair` | `probe5a`（AIC 空转） |
|---|---:|---:|
| Task Duration | 11,876.9 | 12,244.7 |
| AIC `aic_time` | 10,169.7 | 10,488.9 |
| AIC `cube` | **263.7** | 0 |
| AIC `mte1/mte2/fixpipe` | 366 / 733 / 823 | ~0 |
| AIC `scalar` | 1,303.7 | 144.9 |
| AIC `scalar_wait_id0`（等 AIV 的 ready） | 4,591.7 | 5,256.7 |
| **AIV `aiv_time`（跨度）** | **10,168.4** | **10,488.8** |
| AIV `vec` | 7,681.0 (64.7%) | 7,761.0 (63.4%) |
| AIV `scalar` | 4,311.7 (36.3%) | 4,553.3 (37.2%) |
| AIV `scalar_vector_stall` | 5,536.5 | 5,567.8 |
| AIV `scalar_wait_time` | 865.4 | 905.9 |

两条关键读数：

1. **AIV 的跨度（10.17 ms）≈ 整个 kernel 的时长（11.88 ms）**，而 AIC 的 cube 只占
   0.26 ms、全部 pipe 加起来也只有 ~3.2 ms ⇒ **AIC 的活在 AIV 的时间线里被吃掉了**，
   摘掉它只会让流水更"空转"（+370 µs），收益为 0。
2. AIV 侧 `vec` 7.68 ms 常年占 65%，`scalar` 4.31 ms、`scalar_vector_stall` 5.54 ms
   ⇒ **真正的瓶颈是 AIV 的向量指令流**（标量停在向量队列上说明向量单元是满的）。

## 4. 这对 stage 重设计意味着什么

### 3.1 跨用例复核：三种形状都是 AIV 跨度 ≈ kernel 时长

同一批 `leafpair` 包、`msprof op` 整核画像，三个结构完全不同的 perf case：

| perf case | 形状 | Task Duration (µs) | **AIV 跨度 (µs)** | 跨度/时长 | AIV `vec` | AIC `cube` |
|---|---|---:|---:|---:|---:|---:|
| 63 | `B1 HK96 HV96 T8192` | 11,876.9 | 10,168.4 | **86%** | 7,681.0 (64.7%) | 263.7 (2.2%) |
| 65 | `B1 HK8 HV24 T32768`（GVA，`HK<HV`） | 11,003.0 | 10,501.2 | **95%** | 8,001.0 (72.7%) | 274.7 (2.5%) |
| 59 | `B4 HK96 HV96 T128`（小 T、多 batch） | 1,613.1 | 1,549.7 | **96%** | 1,014.6 (62.9%) | 17.4 (1.1%) |

三个用例的结论一致：**AIC 的 cube 占比都在 1~2.5%，AIV 的跨度就是 kernel 的时长**
⇒ "AIC 有余量"在 A2/A3 上反映不了成收益。

1. **stage 边界照旧 8 个**：`C2→V3→C4→C5→C7` 是数据依赖强行串起来的，
   而 V0/V1/V6 都在 AIV 上、复用同一段私有区 ⇒ 没有重切边界的证据。
2. **"把 AIV 的贵段外迁到空闲的 AIC"这条路线整体作废**：
   - A4′（叶子逆外迁 cube）实测 **+2,516 µs > 收益 2,362 µs**（见
     `kda_prepare_a2a3_A4_prime_probe_result.md`）；
   - 反过来把 C4/C5 的小矩阵乘搬到 AIV 也没有意义（AIC 本来就藏得住）。
   ⇒ **AIC 余量不是可用并行度**（核间握手是逐回合 lockstep）。
3. **`design_a2a3_e2_stage_attribution.md` 的"分 Stage 屏蔽差分"表不能当成本分解用**
   （各项之和 > 100%，且与本节直接测量矛盾：屏蔽 C2 在那一版记 −1,988 µs，
   本节同样屏蔽 C2 是 +20 µs）。屏蔽一个 stage 同时会拿掉它的内部 wait/set 与
   指令发射压力，读数只能当"上界提示"。**要用它做决策时必须用本节这种"整侧空转"的对照。**
4. ⇒ **重设计的着力点只剩 AIV 内部**：减少向量指令条数与标量↔向量耦合。
   已量化、可做的候选见下表。

## 5. 修正后的着力点（全是 AIV 侧，含实测上限）

| 优先级 | 项 | 实测上限 | 依据 | 改动面 |
|---|---|---:|---|---|
| **P1** | 叶逆里"每步一次 `Gather`"（62 次/chunk-head） | **1,264 µs（10.6%）** | 只关掉每步 Gather 的屏蔽实测 | `V3Vf`/`SolveTwoLeavesRightLooking` 内部 |
| P2 | V1 三族指数与 V0 的 gate 指数重复计算 | **1,460 µs（12.0%）** | 屏蔽 V1 的 tile 循环 | 涉及 V0/V1 的 UB 驻留（要不要常驻 `E(G)`） |
| P3 | V6（qg/kg/K_beta_g/V_beta） | 2,002 µs（16.4%） | 块级屏蔽 | 已是块指令，需要新思路 |
| P4 | V0 之外其它逐行/标量往返 | 由 `scalar` 4.31 ms / `scalar_vector_stall` 5.54 ms 指示 | 整核画像 | 需逐段定位 |
| — | A4′（外迁 cube）、AIC 侧调度（回合 3→2 等） | **≈0 或负** | 本文 §2 + A4′ 文档 | 不建议 |

## 6. 产物

```text
补丁: patch_probe5.py（variant a=全 AIC 空转 / c=只空转 C2）
tag : probe5a / probe5c
画像: out/p5a/（AIC 空转） out/lp/（leafpair 基线）的 PipeUtilization.csv
```
