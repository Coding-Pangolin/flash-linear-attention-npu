# G1 — USE_SYNC_PLAN_V1 窗级 C_S0/C_S1（reject）

> 日期：2026-07-29  
> 基线 G0：**3789 µs**

## 改动

- 窗内两 head 算完再 **一次** Set/Wait `C_S0` / `C_S1`
- 保留 per-head `V_GATE`/`C_S2` 与 per-head `V_S0`
- 代码保留在 `USE_SYNC_PLAN_V1`

## 验收

| 项 | 结果 |
|----|------|
| suite | **green**（含 split）|
| Task Dur | **3940 µs (~3.94 ms)** |
| Δ vs G0 | **+0.15 ms** |
| 裁决 | **reject** → `USE_SYNC_PLAN_V1=0` |

## 根因（简）

窗级合并推迟 `Stage0Vec(h0)` / `Gate(h0)`，吃掉 Prefill soft-lead 与 **S1(h1)∥Gate(h0)** 重叠。  
nBk=1 后「每 head 每 stage ≤1 Set」**已满足**；再跨 head 合并与 DESIGN 并行契约冲突。

## 下一刀

G2：`USE_FOLD_BAR_SLIM`（V 链末 BAR），不改 CrossCore 次数。
