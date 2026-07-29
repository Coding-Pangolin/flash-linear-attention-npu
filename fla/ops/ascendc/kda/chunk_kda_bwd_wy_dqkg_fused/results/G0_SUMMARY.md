# G0 — 重钉（F3a' / BK128 后）

> 日期：2026-07-29  
> 对照：F3a' 3861 µs；旧 sim P1a（BK64 时代）WAIT 27.6% / BAR 30.4%

## 板端

| 项 | 值 |
|----|-----|
| Task Duration | **3789 µs (~3.79 ms)** |
| Block Dim | 20 / Mix 40 |
| Prof | `results/prof_g0/OPPROF_20260729175033_LOCYNWOOKPPZRCYY` |
| e2e host avg | ~31 ms（含 warmup 外 iters）|

相对 F3a' 3861：**−72 µs**（噪声带内），确认基线 **~3.8 ms**。

## Sim T1024

Out: `results/prof_sim_g0/OPPROF_20260729175116_HSAZDCCPMSTTJXDE`  
Total tick：**506,446**（旧 P1a 1,392k；BK128 + 默认栈后大幅下降）

| | P1a 旧 sim | **G0（BK128）** |
|--|------------|-----------------|
| AIV BAR | 30.4% | **22.0%** |
| AIV WAIT_FLAG(+DEVI) | ~8.5%+ | **14.1%+4.9%** |
| AIV MOVEMASK | 11.9% | **7.3%** |
| AIC WAIT_FLAG(+DEV) | 27.6% DEV | **35.2%+15.7%** |
| AIC BAR | 22.7% | **14.8%** |
| AIC ND2NZ | 14.1% | **2.8%** |

BK128 已大幅砍 ND2NZ；**Cube 仍以 WAIT 为主**（合计 ~51%）。G1 跨 head 合并已证伤 overlap → 下一刀 G4 应盯 **Stage0/3 A 复用以外的等待源**，或 Op 切分占用。
