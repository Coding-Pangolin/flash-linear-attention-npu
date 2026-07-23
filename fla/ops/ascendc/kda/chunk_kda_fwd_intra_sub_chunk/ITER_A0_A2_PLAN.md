# 迭代优化 Plan：A0 → A2（借 PR190）

> 基线：**P1b ≈ 3.705 ms**  
> 总析：`PR190_BORROW_ANALYSIS.md`  
> 门禁：改码 → device7 精度 → device1 msprof → ΔDur ≥ **0.05 ms** 保留，否则 default off + commit  
> 约束：不融合 · 910B · 单变量

---

## 总览

```text
A0  UB g/beta chunk resident     ← 本轮实现
A1  L1 scratch/resident 分区文档  ← 工程债，可与 A0 同 commit
A2  Score MMAD1 ‖ 搬 W / L0 加深  ← A0 门禁后
A3  GVA hk 弱复用                 ← 有证据再开
A4  交错 window + credit          ← 默认不做
```

---

## A0 — `USE_UB_G_BETA_RESIDENT`

### 问题

`g` / `beta` 在 Prep→WriteSolve→… 链上**不被改写**，却每个 iSub 从 GM 重载（`PrepareSub` 的 `CopyVectorIn(g)`、`LoadBetaRows`、`LoadMidRow`）。

### 方案

```text
ProcessChunkAiv 入口（ResolveChunk 后）:
  Load g[bos : bos+localT, K] → gResBuf_   (dtype T, ≤ MAX_BT×MAX_K)
  Load beta[bos : bos+localT] → cast fp32 → betaResBuf_

PrepareSub(iSub):
  g 切片 / mid 行 ← gResBuf_（Local←Local），不再 GM
WriteSolve(iSub):
  beta[iTi:iTi+valid] ← betaResBuf_，不再 LoadBetaRows(GM)
```

### 实现要点

| 项 | 选择 |
|----|------|
| 宏 | `USE_UB_G_BETA_RESIDENT=1` 试验 |
| `MAX_BT` | 128（tiling 允许 32/64/128） |
| UB | `gResBuf_`: 128×128×sizeof(T)≈32KB；`betaResBuf_`: 128×4=512B；**独立于 `vecBuf_`** |
| dual-AIV | V1：两核各自加载完整 resident（简单正确）；后续可改成 sub0 加载 + barrier |
| varlen | 按 `localT` 加载，索引用 chunk 内相对 tok |
| mid | 从 `gResBuf_[localTok*K]` 取一行再 Cast→`midBuf_` |

### 风险

- UB 总占用上升（~32KB）；确认不挤爆 192KB（当前 arena ~48KB + 其它 ~几 KB → 仍裕量）。  
- Local←Local `DataCopy` 对齐：整行 `K*sizeof(T)` 须满足现有 Copy 条件或走 Pad。  
- 单刀或然 Δ&lt;0.05（砍 AIV MTE2，墙钟仍可能钉 AIC）。

### 验收

精度 suite + msprof vs 3.705；记 `aiv_mte2` 是否下降。

---

## A1 — L1 分区文档化（无性能门禁）

### 方案

在 kernel 注释写死布局：

```text
[0, SCORE_L1A)           Score L1A (QG/W)
[SCORE_L1A, MCH_L1_BASE) Score L1B (kneg)
[MCH_L1_BASE, …)         MCH I/X/Y/L (+ 预留 RESIDENT)
```

禁止 Score/MCH 互踩；为后续真驻留留命名空间。可与 A0 同批提交。

---

## A2 — Score：MMAD1 ‖ 搬 W（A0 之后）

### 方案（示意）

```text
CopyGmToL1B(kneg); CopyGmToL1A(qg)
L1→L0; Mmad(qg,kneg) → Fix Aqk
  ∥ 期间 MTE2 搬 W → L1A
复用 L1B; Mmad(w,kneg) → Fix Akk
```

单变量宏；事件 id 避开 `MCH_EVT`。仅当 Score 段仍见 MTE2/空窗时值得做。

---

## A3 / A4（备忘）

- **A3 GVA hk**：同 `(B,chunk,hk)` 相邻时少载 K。  
- **A4 window+credit**：仅 A0–A2 后空窗仍厚；高成本。

---

## 明确不做

StoreAqk 空窗、纯 T5 stage 批、无证据 DEPTH=3、L0C→L1。

---

## 年表

| 阶段 | 状态 |
|------|------|
| Plan 本文 | done |
| A0 实现 + 门禁 | **running** |
| A1 分区注释 | with A0 |
| A2 | pending |

---

## A0 门禁结果（2026-07-23）

| 项 | 结果 |
|----|------|
| 实现 | 已合入（chunk 级 g/beta resident + `gBetaResidentActive_` 回退） |
| A1 L1 分区注释 | 已写入 `MCH_L1_BASE` 旁 |
| 精度 | 短 seq `nTok<BC` 曾 `akkd_rel=1`；加 `nTok<bc_` 回退后出现 **aicore exception** |
| 结论 | **default `USE_UB_G_BETA_RESIDENT=0`**；代码保留待修 Local←Local 同步 / 短块索引 |
| 基线 | 仍为 P1b **3.705 ms** |

### 已澄清的设计点（下次重开必守）

1. Resident 范围是 **当前 chunk**（`localChunk*bt_ .. +nTok`），不是 dense 的 `localT=t_`。  
2. Prep/WriteSolve 索引用 **chunk-local** `iSub*bc_ + row`。  
3. `gRes` 容量 `MAX_BT * G_RES_MAX_K`（K≤128）以免与 `vecBuf_(MAX_K=256)` 挤爆 UB。  
4. 短块 `nTok < bc_` 与 dual-AIV pad 行需单独验，或继续 GM 回退。
