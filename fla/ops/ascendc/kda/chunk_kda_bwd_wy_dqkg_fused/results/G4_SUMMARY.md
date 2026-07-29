# G4 — USE_WY_L1_RESIDENT_V2（reject）

> 日期：2026-07-29  
> 基线 G0：**3789 µs**；sim ND2NZ 已仅 **2.8%**（策略写 14% 已过时）

## 试验

| 变体 | 做法 | Task Dur | Δ |
|------|------|----------|---|
| G4a | per-head Stage2→Wait Mask→Stage3 + L1 park | **3914 µs** | **+0.13 ms** |
| G4b | 恢复批量调度；Stage3 Preload A→高 L1 + gemm2 skipLoadA | **3820 µs** | **+0.03 ms** |

## 正向修复

- G4a 回归根因：**Wait Mask 串在两 head Stage2 之间**，丢并行；非 ECC/精度问题。  
- G4b 去掉串调度后精度仍绿，但 ND2NZ 已低，显式 Preload+skip **不优于** 原 gemm2 内联 load → flat。

## 裁决

`USE_WY_L1_RESIDENT_V2=0`（代码保留：`l1AByteOff`、Preload offset、Stage3 V2 路径）。  
见 `G4_L1_PEAK.md`。

## 下一刀

G5 Gate/Epilog fold 审计，或针对 Cube **WAIT~51%** 另开 sync 刀（勿再跨 head 合并 C_S*）。
