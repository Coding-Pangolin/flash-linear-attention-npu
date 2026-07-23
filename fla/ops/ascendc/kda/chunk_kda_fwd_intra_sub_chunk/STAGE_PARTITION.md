# ChunkKdaFwdIntraSubChunk：Stage 划分与优化方向

> 对标 `ref_pr190/a2_a3_common_optimization_notes.md` §1（数据依赖切 Stage / 2-head window）。  
> 范围：tiling key 1 · `MIX_AIC_1_2` · 默认热路径（Score Tile + L0 ACC Dual + L1 resident）。  
> 关联：`CUBE_PIPELINE_DESIGN.md`、`AIV_MCH_IDLE_PLAN.md`、`SCORE_TILE_CROSSCORE_PLAN.md`、`L0_ACC_MCH_DESIGN.md`、**`PR190_BORROW_ANALYSIS.md`**（可借鉴项总表）

**Vector = AIV，Cube = AIC。**

---

## 1. 切 Stage 的原则（摘自 notes）

1. 同一 stage 内尽量放只依赖原始输入、或依赖已在本 stage 入口 ready 的计算。
2. 跨 Cube/Vector 的中间结果写 GM workspace；producer 在 stage 末尾 set flag，consumer 在下一 stage 入口 wait。
3. wait 放在消费者 stage 入口，放在 row/tile 内循环外。
4. stage0 理想情况：两侧互不等，各自生产。
5. 一个 `(chunk, hv[, iSub])` 在所有 stage 中命中同一 workspace slot。
6. 对独立单元（notes：2-head；本算子：iSub / task）优先 **按 stage 成组推进**，而非单单元跑完再下一个——但须保留「长 stage 与短工作重叠」（见 §5）。

本算子独立调度单元：

| 层级 | 单元 | 依赖 |
|------|------|------|
| 分核 task | `(B, HV, NT chunk)` | **互不依赖** |
| 核内 | `iSub = 0..NC-1`（BC×BC 块） | **互不依赖** |
| 单 iSub 内 | Prep → Score → WriteSolve → MCH → Store | **强 CV 链** |

---

## 2. 核心依赖图（单 `(chunk, hv, iSub)`）

```text
原始输入:
  Q, K, G, beta

stage_0:
  Vector:
    mid_g ← g[mid]
    QG    = Q * exp(g - mid)      → scoreWs[QG]
    W     = K * exp(mid - g) * …  → scoreWs[W]
    KG    = K * exp(g - mid)      → scoreWs[KG]
  Cube:
    （空）                         // 等 sync_ready
                                   // 对比 prepare_wy：其 stage0 Cube 有 Dkbg/Dvb/KKT

stage_1:
  Cube:
    Aqk = QG @ KGᵀ * scale        → cmat / 中间平面
    Akk = W  @ (−KG)ᵀ             → cmat[AKK]
  Vector:
    （空 / WaitDone）

stage_2:
  Vector:
    Aqk' = tril(Aqk)              → aqkBuf_（供 Store）
    L    = f(tril(Akk), β, …)     → cmat[AKK]
    X0   = I - L                  → SOLVE_X
  Cube:
    （空 / WaitSolveReady）

stage_3:
  Cube:
    X = (I - L)⁻¹                 // Neumann L0 ACC Dual → SOLVE_X
  Vector:
    Prep(i+1) + ready             // 已与 MCH 重叠（默认路径）
    // 可选：StoreAqk（不依赖 SOLVE_X）— 见 §6 P0

stage_4:
  Vector:
    aqk  ← Cast(aqkBuf_) → GM     // ★ 数学上不依赖 stage_3
    akkd ← SOLVE_X → GM           // 依赖 stage_3
  Cube:
    （空）
```

### 与 prepare_wy 的关键差异

| | prepare_wy | 本算子 |
|--|------------|--------|
| stage_0 | Vector **与** Cube 同时有活 | 仅 Vector；Cube 空等 |
| 同 stage 两侧 | 常「一侧产、一侧在同 stage 末消费」 | 多为「一侧干、一侧整 stage 空等」 |
| 成组对象 | 2-head window | 可成组对象是 **独立 iSub / task** |
| 最长 stage | 各 stage 相对均衡 | **stage_3 MCH ≫ stage_1 Score** |

气泡根源：四拍 CV 交替里，多数拍只有一侧在干活。

---

## 3. 同步点

| 同步点 | 方向 | Producer 产物 | Consumer |
|--------|------|-----------------|----------|
| sync_ready | Vector → Cube | `QG / W / KG` | Cube stage_1 |
| sync_done | Cube → Vector | `Aqk / Akk` | Vector stage_2 |
| sync_solveReady | Vector → Cube | `L, X0` | Cube stage_3 |
| sync_solveDone | Cube → Vector | `X=(I−L)⁻¹` | Vector stage_4（**仅 akkd**） |

代码落点（`op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp`）：

- Vector 主循环：`ProcessChunkAiv`
- Cube 主循环：`ProcessChunkAic`
- slot：`iSub % depth_`（`SCORE_QUEUE_DEPTH=2`）

---

## 4. 当前主循环骨架（按 sub 串完）

等价于 notes 所说「单 head 全流程跑完再下一个」：

```text
AIV:
  Prep(0); Set(ready)                          // stage_0(sub0)
  for i in 0..NC-1:
    Wait(done)                                 // sync_done
    WriteSolve(i); Set(solveReady)             // stage_2
    if i+1 < NC: Prep(i+1); Set(ready)         // 下一拍 stage_0 ‖ 本拍 stage_3
    Wait(solveDone)                            // sync_solveDone
    Store(aqk + akkd)                          // stage_4 整包

AIC:
  for i in 0..NC-1:
    Wait(ready); Score; Set(done)              // stage_1
    Wait(solveReady); MCH; Set(solveDone)      // stage_3
```

时间线：

```text
         stage_0      sync     stage_1     sync    stage_2      sync        stage_3         sync     stage_4
AIV:  [ Prep QG/W/KG ]─ready─[ 空等 Done ]─done─[ WriteSolve ]─sReady─[ Prep下一 / 空等 ]─sDone─[ Store ]
AIC:  [     空等      ]───────[ Score×2  ]──────[    空等     ]───────[   MCH Dual    ]──────[  空  ]
```

Workspace：`(chunk, hv)` 任务内，`iSub` 使用 `slot = iSub % 2`；同一 iSub 的 Score/MCH/Solve 平面必须同 slot（notes 原则 5）。

---

## 5. 「按 stage 成组」长什么样（示意，不可照搬）

notes 的 2-head window：

```cpp
for (head in window) stage_0(head);
for (head in window) stage_1(head);
for (head in window) stage_2(head);
for (head in window) stage_3(head);
```

套到本算子两个独立 iSub（DEPTH=2）的字面翻译：

```cpp
// 仅示意 —— USE_S2C_BATCH 实测更慢，默认关闭
for (j in window) Prep(j), Set(ready);           // stage_0 ×n
for (j in window) Wait(ready), Score, Set(done); // stage_1 ×n
for (j in window) Wait(done), WriteSolve(j);     // stage_2 ×n
barrier;
for (j in window) Set(solveReady);
for (j in window) Wait(solveReady), MCH, Set(solveDone); // stage_3 ×n
for (j in window) Wait(solveDone), Store(j);     // stage_4 ×n
```

**为何失败**：成组推进拆掉了默认路径里的 **Prep(i+1) ‖ MCH(i)**。本算子 stage_3 远长于 stage_1，批处理会让 AIV 在整波 MCH 前空等、AIC 在整波 Score 后空等。

若要成组，应做 **交错窗口 + flag credit**（允许 producer 有限领先），而不是「整 stage 批处理」。DEPTH=3 只加 Prep 多半只是挪空等位置（见 `AIV_MCH_IDLE_PLAN.md`）。

---

## 6. 优化方向

### P0 — 拆 stage_4：`StoreAqk ‖ MCH`（改依赖边界）

```text
目标:
AIV: solveReady → Prep(i+1) → StoreAqk → WaitSolveDone → StoreAkkd
                                 └─ 填原 Wait 空窗 ─┘
```

- 把不依赖 `SOLVE_X` 的 aqk 回写挪进 stage_3 的 Vector 侧。
- 宏：`USE_STORE_AQK_UNDER_MCH`（实现保留）。
- **实测**：median **3.674** vs P1b **3.705**（Δ−0.031 &lt; 0.05 ms 门禁）→ **default off**。
- 结论：方向正确，当前画像下墙钟仍钉 AIC Fixpipe/MTE2，AIV 尾前移露不出 ≥0.05 ms。

### P1 — 否决：纯 stage 批处理（T5 / `USE_S2C_BATCH`）

- MMAD×DEPTH → MCH×DEPTH：Dur **5.36 &gt; T4 4.11**。
- 代码保留，默认 **0**。不要再当主路径。

### P2 — 交错多拍 / DEPTH≥3 + credit flag

- 对标 notes §5 window + §9 raw flag 背压：扩大「同一方向未消费 Set」额度。
- 可隐藏 Store / prologue，但需 redesign flag（非简单 1:1）。
- 仅当 profile 证明 `WaitSolveDone` 空窗仍厚且 P0 类手段不够时再开。

### P3 — 跨 task 的 2-chunk / 2-hv window

- 同一核上两个独立 `(B,HV,chunk)` 按 stage 交错。
- 代价：workspace ×2、slot/flag 协议、与现 DEPTH=2 ping-pong 叠加复杂。
- 收益偏摊 prologue / 藏 Store；实现成本高于拆 Store。

### P4 — GVA `hk` 级 Prep 复用（对标 notes §6 KKT cache）

- `iH = iHv / group_`，同 chunk 多 hv 共享 K。
- 若调度让同 `(B, chunk, hk)` 相邻，可 resident K / mid-row，减重复 `PrepareSub` 载 K。
- 属 group 内弱共享，不是跨 chunk 数据依赖。

### 已落地、与 Stage 相关的优化（不必再挖同构刀）

| 项 | 状态 |
|----|------|
| Score Tile + L1B(kneg) | on |
| MCH L0 ACC Dual | on |
| MCH L1 resident（I） | on |
| `USE_S4_NO_POST_BARRIER` | on（去 WriteSolve→solveReady 双 AIV barrier） |
| Prep(i+1) ‖ MCH(i) | 默认路径已有 |
| S2b steal MMAD | 试过无优，off |
| soft-prefetch KG | 试过无优，off |

---

## 7. Checklist（迁移 notes 到本算子时）

1. 每个中间张量：producer / consumer / pipe / 内存层级是否列出。
2. stage 边界是否只在真实跨侧依赖处。
3. wait/set 是否在 stage 入口，而非 row 内反复 CrossCore。
4. 同一 iSub 的 score/cmat/solve 是否始终同 `slot`。
5. 成组推进是否保留「长 MCH ‖ 短 Vector」重叠；禁止为对齐 notes 骨架而拆掉有效重叠。
6. 扩大 window / DEPTH 时是否重做 flag credit（notes §9 / checklist 12）。

---

## 8. 一句话结论

本算子应记为 **5 段 CV 链（Prep / Score / WriteSolve / MCH / Store）× NC 串行**；与 notes 最大差距是 **多数 stage 只有一侧有活**，以及 **独立 iSub 未按 stage 成组**。可优化方向按优先级：拆无依赖的 StoreAqk（已试、门禁下关）→ 有证据再上交错 window/credit → GVA hk Prep 复用；**不要**再走纯 MMAD 批再 MCH 批的 T5 形态。
