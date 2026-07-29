# G2 — USE_FOLD_BAR_SLIM（reject / flat）

> 日期：2026-07-29  
> 基线 G0：**3789 µs**（SYNC_PLAN=0）

## 改动

开启已有宏位 `USE_FOLD_BAR_SLIM=1`（Gate ColSum / Epilog Mul / state fold 链末 BAR）。

## 验收

| 项 | 结果 |
|----|------|
| suite | **green** |
| Task Dur | **3802 µs** |
| Δ vs G0 | **+0.01 ms**（flat，未过 −0.05）|
| 裁决 | **reject** → 宏回 **0**，代码保留 |

## 下一刀

G1/G2 均未吃到 CrossCore/BAR 墙钟。下一优先 **G4 WY L1 resident v2**（ND2NZ 14% vs PR190 4.6%），或重开 G1 为「仅减无 overlap 的冗余」审计（勿再跨 head 合并 C_S1）。
