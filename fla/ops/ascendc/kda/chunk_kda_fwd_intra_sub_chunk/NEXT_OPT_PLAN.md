# IntraSubChunk 下一轮优化 Plan（post-P1b）

> 当前保留最佳：**P1b ≈ 3.705 ms**（Score Tile + MCH Dual/L1 resident + S4a + zN single-load）  
> 约束：不融合；910B；单变量门禁（精度 → msprof → ΔDur ≥ 0.05 ms 才默认开）

## 已关实验（代码保留）

| 实验 | 结果 |
|------|------|
| MCH_ITERS_2 / SKIP_XI / FIX_OVERLAP | 精度败或 Dur 平 |
| Prologue `L@L ∥ Load X/I` | 精度 OK，Dur −0.026 → off |

## 方案路线（按期望收益排序）

### P0 — `USE_STORE_AQK_UNDER_MCH`（见 `AIV_MCH_IDLE_PLAN.md`）

**状态：已试 · 门禁失败 · 代码保留默认 0**  
精度 OK；Dur **3.674** vs **3.705**（−0.031 &lt; 0.05）。

### P1 —（备选）DEPTH=3 Prep(i+2) — **默认不做**

期望 ≪0.05 ms；WS×1.5。仅当有新仿真证据再开。

### 当前瓶颈判断

墙钟钉在 **AIC Fixpipe + MTE2**（~0.9 ms + ~0.8 ms pipe time）。AIV 侧再叠工作（prologue / StoreAqk）实测都 &lt;0.05 ms。下一步若要动墙钟，需 **缩短 MCH Fixpipe/Nd2Nz 本身**（算法或硬件路径），而不是再填 AIV 空窗。

### 下一刀（借 PR190，见 `PR190_BORROW_ANALYSIS.md`）

| 优先级 | 项 |
|--------|-----|
| **A0** | UB：`g`/`beta` chunk resident（跨 Prep/WriteSolve 无改却每拍重载） |
| A1 | L1 scratch/resident 显式分区（热路径只用 ~十几 KB ≪ 512KB） |
| A2 | Score：MMAD1 ‖ 搬 W / L0 加深 |
| — | 勿再纯 stage 批 / 勿再填 WaitSolveDone |

### 不做

- DEPTH=3 / 双 Set 同 flag
- L0C→L1（非 910B）
- 再砍 Neumann iter（精度败）
- 无新证据重开已否实验
