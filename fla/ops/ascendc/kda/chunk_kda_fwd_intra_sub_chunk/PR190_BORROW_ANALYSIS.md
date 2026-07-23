# 从 PR190 可借鉴的流水优化（对照本算子）

> 对照：`ref_pr190/a2_a3_common_optimization_notes.md`（prepare_wy_repr_bwd）  
> 本算子现状：`STAGE_PARTITION.md` · 热路径 P1b ≈ **3.705 ms**  
> 约束：不融合 · 910B · 单变量门禁（ΔDur ≥ 0.05 ms）  
> 目的：把 PR190「切 stage / stage 内双 buffer / 预搬运 / stage 间 GM slot」映射到 `ChunkKdaFwdIntraSubChunk`，标清 **已有 / 可试 / 勿照搬**。

---

## 0. 一句话

PR190 流水好，是因为 **两侧 stage 都有活 + L1/UB 跨 stage 真驻留 + stage 内双 buffer**；本算子多数 stage 只有一侧干活，墙钟钉在 **AIC Fixpipe+MTE2**。  
可借鉴的不是再「填 AIV 空窗」（已试失败），而是 **把 512KB L1 / UB 余量用在跨 iSub、跨 Score↔MCH 的真驻留**，以及 **小向量 g/beta 的 chunk 级 resident**。

---

## 1. PR190 手法 → 本算子映射总表

| PR190 条款 | 本算子现状 | 可借鉴度 | 备注 |
|------------|------------|----------|------|
| §1 依赖切 Stage | 已有 5 段 CV 链文档化 | ★★ | 结构已对齐；勿为对齐而拆 Prep‖MCH |
| §2 Cube L1 resident | Score L1B(kneg)+MCH I 已驻留；**L1 用量 ≪ 512KB** | ★★★ | **最大未用杠杆** |
| §3 Cube L0 双 buffer | MCH Dual 已有；Score 仍偏串 | ★★ | Score 侧 L0 ping/pong / 与 MTE2 重叠可加深 |
| §4 Vector 双 buffer | Prep 有 tile 环；input/output 未系统 ping/pong | ★★ | 与 g/beta resident 一起做更干净 |
| §5 2-head + 4 GM slot | DEPTH=2 slot 已有；2-iSub 成组 = T5 已败 | ★ | 勿再纯 stage 批；要做须 credit flag |
| §6 KKT 按 hk 缓存 | 无等价物 | ★★ | GVA：同 `(chunk,hk)` 的 K / Prep 弱共享 |
| §7 V=256 K 分块 | BC=16 无此问题 | — | 不适用 |
| §8 物理转置避散写 | Store 已连续半行 | ★ | 已够用 |
| §9 Raw flag 背压 | CrossCore 1:1 | ★ | 扩 window 才需要；当前非瓶颈 |
| §10 **beta/g/exp(g) UB resident** | **每 Prep/WriteSolve 重载** | ★★★ | **小 shape、跨 stage 无改 → 长期驻留** |
| §11 Vector row tiling | PrepMaxTileRows 已有 | ★ | 可复核 16KB 预算是否过松/过紧 |
| §12 模板 TilingKey | 现有 key 较粗 | ★ | 非性能主刀 |
| §13 Chunk 分核 HV 串行 | 已按 `B×HV×NT` 分核 | — | 一致 |

已试、**勿再当主刀**（详见 `NEXT_OPT_PLAN.md` / `AIV_MCH_IDLE_PLAN.md`）：

| 实验 | 结果 |
|------|------|
| StoreAqk ‖ MCH | −0.031 ms |
| MCH prologue `L@L∥X/I` | −0.026 ms |
| S2C 纯 stage 批 | 回归 5.36 ms |
| soft-prefetch / WS half / Prep-before-done | 平或 NaN |

---

## 2. L1 预算：512KB 远未用满

### 2.1 当前 footprint（模型 shape：`BC=16, K=128, T=bf16/fp32`）

```text
Score 区（MCH_L1_BASE 之前）:
  L1A(QG/W) + L1B(kneg)  ≤ 2 × BC×K×sizeof(T) = 2 × 16×128×2 = 8 KB

MCH 区（USE_MCH_Y_SINGLE_LOAD=1）:
  I / X / Y / L 各 BC×BC×fp32 = 1 KB → 共 4 KB
  （无 alias 时最多约 7 KB）

合计热路径 L1 ≈ 十几 KB 量级  ≪  512 KB
```

PR190 把 L1 拆成 **scratch ping/pong + resident ping/pong**，跨 stage 复用 `K/A/DW/DU`。  
本算子已有「局部驻留」，但 **没有「跨 stage / 跨 iSub 的 resident 分区」**，余量基本闲置。

### 2.2 可驻留什么（按依赖，不是按「看着大」）

| 候选 | 生命周期 | 跨谁复用 | 期望 | 风险 |
|------|----------|----------|------|------|
| **I（已做）** | chunk 内 MCH | 各 iSub MCH | 已落地 | — |
| **kneg / KG（已做半截）** | 单 iSub Score 两拍 MMAD | MMAD1→MMAD2 | 已落地 | — |
| **L 在 MCH 内（已做）** | 单次 MCH | Neumann 轮次 | 已落地 | — |
| **Score 产出后、WriteSolve 前的 cmat 中间面** | 单 iSub | 无：AIV 立刻消费 | 低 | 不值得 |
| **同 hk 的 K（GVA）** | 同 chunk 多 hv | Prep/Score 载 K | 中 | 调度须同 hk 相邻；AIV 侧为主 |
| **扩展 Score L1：QG 驻留 + W 与 MMAD1 重叠搬** | 单 iSub | 已部分有 | 低–中 | 事件与 `MCH_EVT` 错开 |
| **MCH 中间 X/Y 少进 GM** | 单次 MCH | 910B 无 L0C→L1 | 高但受限 | 已用 TMP 回灌；再砍 Fixpipe 需证据 |

**推荐主刀（Cube L1）：** 在现有 `MCH_L1_BASE` 之上显式划 **resident 区**（对标 notes §2），不要继续「所有 L1 都当 scratch」。优先验证：

1. **Score L1 分区文档化**：`[0, ScoreA) [ScoreA, ScoreB) [MCH_BASE, …)`，禁止 MCH/Score 互相踩。  
2. **同 Resource 生命周期**：chunk 级 `sharedResource` 已有；检查 Score 是否仍隐式重建/踩 MCH resident。  
3. **GVA**：同 `(B,chunk,hk)` 连续时，K 行或 mid 相关载入是否可少做一遍（偏 AIV，见 §3）。

> 不要指望「把 L1 填满」本身加速；只有 **去掉重复 Nd2Nz/Fixpipe** 才动墙钟。余量是允许做真驻留的前提，不是目标。

---

## 3. UB：g / beta（及派生量）跨 stage 无改 → 长期驻留

对标 PR190 **§10 beta/g/exp(g) Vector Resident**。

### 3.1 现状（每次重载）

| 数据 | 谁用 | 现在 | 大小（BT=64,K=128,bf16） |
|------|------|------|---------------------------|
| `g` 行块 | Prep（每 iSub） | `PrepareSub` 内 `CopyVectorIn(gT)` | 每 tile；整 chunk ≈ `BT×K×2` = **16 KB** |
| `mid = g[mid]` | Prep | 每 iSub `LoadMidRow` | **256 B** |
| `beta` | WriteSolve | 每 iSub `LoadBetaRows` | 整 chunk **128 B**（BT×fp32）或仅 BC |

`g`/`beta` 在 **Prep → Score → WriteSolve → MCH → Store** 链上 **不被改写**；变的是 iSub 切片下标。  
这正是 notes「小向量独立 ping/pong / resident，不挤 matrix arena」的适用前提。

### 3.2 目标形态（示意）

```text
chunk 入口（或 task 入口）一次:
  Load g_chunk  → ubGResident_     // BT×K 或按 dual-AIV 半 chunk
  Cast → fp32 可选；或按需切片再 Cast
  Load beta_chunk → ubBetaResident_ // BT

PrepareSub(iSub):
  切片 g[iTi : iTi+BC]（或 live 行）← resident，不再 GM→UB
  mid 仍按 iSub 从 resident 取一行（或预计算 mid 表）

WriteSolve(iSub):
  beta[iTi : iTi+valid] ← resident，不再 LoadBetaRows from GM
```

### 3.3 收益与门禁预期

- 砍的是 **AIV MTE2**（prep/write 的 g/beta），不是 AIC Fixpipe。  
- 当前画像：`aiv_mte2 ≈ 0.65 ms`、墙钟钉 AIC；单刀可能 **仍 < 0.05 ms**，但：  
  - 与 Prep 向量化 / 少 scalar 叠加更有意义；  
  - 实现代价低、语义清晰，适合作为 **「对标 PR190 的 UB resident」样板刀**；  
  - 若仿真显示 Prep 段 MTE2 厚，则更值得做。

### 3.4 不做 / 慎做

- **不要**把整 chunk `exp(g)` 无脑驻留除非测过：`BT×K` fp32 exp 表 ≈ 32 KB，挤 `vecBuf_` arena（Prep 已用 `6×elems` float）。优先 **resident 原始 g + 现算 exp**（与 PR190「先驻留再算 exp」同构，但本算子 mid 相对每行，exp 表共享面更窄）。  
- **不要**在 Prep 与 StoreAqk 之间抢 `vecBuf_`（已有约束）。resident 应 **独立 TBuf**，对标 notes 的 beta-g ping/pong 与 matrix 分离。

---

## 4. Stage 间「双深度 + 预备 GM slot」——我们已有什么、缺什么

### 4.1 PR190 模型

```text
window 0: slot0/1    window 1: slot2/3    （4 GM slot）
按 stage 成组推进 2 head；同一 (chunk,hv) 全程同 slot
```

### 4.2 本算子模型

```text
DEPTH=2: slot = iSub % 2
默认环: Prep(i+1) ‖ MCH(i)     ← 已保留的「长 stage ‖ 短工作」
T5 成组: Prep×n → Score×n → Write×n → MCH×n → Store×n  ← 已否决
```

| 概念 | PR190 | 本算子该怎么借 |
|------|-------|----------------|
| GM slot 预备 | 下一 head 的 workspace 已分配 | **已有** DEPTH=2；不必为借而扩到 4，除非做 2-task window |
| stage 成组 | 两侧都有活，成组不拆重叠 | **禁止**再上纯 T5；若做窗口必须 **交错 + credit**（`STAGE_PARTITION.md` §5） |
| 入口 wait、环外 sync | wait 在 stage 入口 | **已基本遵守**；勿在 row 内加 CrossCore |

**结论：** 「预备 GM slot」我们已经在用；PR190 的窗口价值来自 **多 head 同 stage 填满两侧**，本算子 iSub 独立但 **MCH 极不对称**，成组会拆掉有效重叠。下刀应在 **驻留与 stage 内流水**，不是再加深无 credit 的 DEPTH。

---

## 5. Stage 内双 buffer / 提前搬运（仍可加深）

对标 notes §3 / §4 / bwd_dhu「Wait 前先搬不依赖数据」。

| 位置 | 已有 | 可加深 |
|------|------|--------|
| Score | L1B(kneg) 驻留 + Tile | MMAD1 期间搬 W；L0A/L0B ping-pong（若仍有 MTE1 气泡） |
| MCH | Dual X∥Y、I 驻留、zN 单载 | Fixpipe 与下一轮 MTE1 重叠（`FIX_OVERLAP` 已试平 → 需新证据） |
| AIV Prep | tile 环、mid 一次/PrepareSub | **g/beta resident**；CopyIn / Cast / CopyOut 槽位化（notes §4 形态） |
| AIC Wait(ready) 前 | soft-prefetch 试过平 | 仅当 Score MTE2 再成瓶颈时重开；优先 L1 分区正确性 |

---

## 6. 建议优先级（可执行）

| 优先级 | 项 | 对标 | 期望 | 门禁策略 |
|--------|----|------|------|----------|
| **A0** | **UB：`g`/`beta` chunk resident**（独立 TBuf） | notes §10 | 降 AIV MTE2；单刀或然 <0.05 | 精度 + msprof；不过则 default off 记账 |
| **A1** | **L1 布局文档 + 断言式分区**（Score / MCH / 预留 resident） | notes §2 | 防踩踏；为后续驻留铺路 | 无性能门禁，属工程债 |
| **A2** | Score：MMAD1 ‖ 搬 W / L0 双缓冲加深 | notes §3 + §4 | 视 Score 段气泡 | 单变量 |
| **A3** | GVA：同 hk 的 K/Prep 弱复用 | notes §6 | 模型若 HV 大更明显 | 需调度假设 |
| **A4** | 交错 window + flag credit | notes §5+§9 | 仅 A0–A2 后空窗仍厚 | 高成本，默认不做 |
| — | 再填 WaitSolveDone（StoreAqk 类） | — | 已证伪 | **不做** |
| — | 纯 stage 批 T5 | — | 已证伪 | **不做** |

当前墙钟主因仍是 **AIC Fixpipe+MTE2**。A0 不直接砍 Fixpipe，但：

1. 对齐 PR190 最可迁移的一条（小向量 resident）；  
2. 减轻 AIV，给后续「Score/MCH 与 Prep 更紧重叠」留空间；  
3. 实现面清晰，易做 ON/OFF 差分。

若只追 ≥0.05 ms 墙钟，需同时盯 **减少 MCH Fixpipe/Nd2Nz 次数**（算法或 910B 允许的回灌路径）——那是另一条线，见 `TARGET_1P5_ANALYSIS.md` / `MCH_SHORTEN_PLAN.md`，与本文「借 PR190 流水结构」互补而非替代。

---

## 7. Checklist（从 notes §14 裁到本算子）

1. 中间张量：producer / consumer / pipe / **GM·L1·UB 哪一层**是否写清。  
2. stage 边界是否只在跨侧依赖；**aqk 与 akkd 依赖是否仍绑在同一 Store**（已拆代码、默认关）。  
3. L1 是否区分 **scratch vs resident**，resident 生命周期是否写明（禁止借 scratch 延寿）。  
4. **512KB 余量**是否有意留给跨 iSub/跨 stage 复用，而不是随便堆临时面。  
5. `g`/`beta`（及可选 `exp`）是否 **跨 Prep/WriteSolve 无改却仍每拍从 GM 搬**。  
6. Vector matrix arena 与 beta-g resident 是否解耦（notes §4+§10）。  
7. 成组 / 扩 DEPTH 时是否保留 **Prep(i+1)‖MCH(i)**；是否重做 flag credit。  
8. 单变量门禁：精度 → device1 msprof → Δ≥0.05 才默认开。

---

## 8. 与现有文档关系

| 文档 | 关系 |
|------|------|
| `STAGE_PARTITION.md` | Stage 现状与「勿照搬成组」；本文补 **L1/UB 驻留缺口** |
| `AIV_MCH_IDLE_PLAN.md` | AIV 空窗刀已闭环；本文不再推同构刀 |
| `SCORE_TILE_CROSSCORE_PLAN.md` | Score/MCH L1 已做部分；本文强调 **预算未用满** |
| `NEXT_OPT_PLAN.md` | 执行队列；建议把 **A0 g/beta resident** 列为下一刀 |

---

## 9. 结论

- **可借：** 依赖切 stage（已做）、**L1 scratch/resident 分区（余量巨大）**、**g/beta UB 长期驻留**、stage 内双 buffer/预搬运加深、GVA hk 弱缓存。  
- **不可借死：** 2-head 式纯 stage 批、无 credit 的 DEPTH 加深、再靠 AIV 填 MCH 空窗刷墙钟。  
- **下一刀首选：** `USE_UB_G_BETA_RESIDENT`（名字待定）— chunk 级加载 g/beta，Prep/WriteSolve 只切片；独立于 `vecBuf_`；按单变量门禁验收。
