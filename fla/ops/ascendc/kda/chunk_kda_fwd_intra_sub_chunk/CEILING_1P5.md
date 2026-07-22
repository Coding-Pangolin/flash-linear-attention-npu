# ChunkKdaFwdIntraSubChunk：1.5 ms 天花板结论

> Shape `(B=1,H=32,T=8192,K=128,BT=64)` bf16 · 空闲卡 device1 · Task Duration 中位  
> 母本：`SCORE_TILE_CROSSCORE_PLAN.md` · 分析：`TARGET_1P5_ANALYSIS.md`

## 1. 最终墙钟（本 sprint）

| 阶段 | Dur med (ms) | 备注 |
|------|-------------:|------|
| Dual 基线 | 4.654 | `USE_MCH_L0_DUAL` |
| T1 Score Tile | 4.627 | L1B kneg |
| T2 CrossCore | 4.402 | Identity 后移 + Score Resource |
| T4 MCH L1 resident | **4.112** | 共享 Resource + I@MCH_L1_BASE + X→TMP |
| T5 S2c（开） | 5.360 | 精度绿；**回退**（默认关） |
| T5 回退后 | ≈4.096 | 回到 T4 带 |

**当前最佳 ≈ 4.1 ms。目标 1.5 ms。缺口 ≈ 2.6 ms（约 2.7×）。**

## 2. 为何单算子内难到 1.5

每核仍跑完整 Prep + Score×NC + MCH×NC + Store，NC=4，task 量大：

- 板端 / 仿真一致：**fixpipe + MTE2 + BAR / CrossCore wait ≫ MAC**
- 910B **无 L0C→L1**；MCH 中间轮仍须 Fixpipe→GM→Nd2Nz（T4 只省 Resource/I 重载与 X 平面污染）
- **S2c 批 MMAD→MCH** 砍了 per-sub barrier，但丢掉 Prep‖MCH 重叠，并强制 tril(aqk) GM spill → **净亏损 ~1.25 ms**
- Prep 已相对 MCH 提前结束（否决再抠 S2a）；S2b steal 已否

粗算：4.1→1.5 需再砍 **~63%** 墙钟。在现有 CV 骨架与 16×16 Neumann×3 工作量下，乐观叠 P1（S4/prefetch ±0.3~0.7）也多半停在 **~3.4–3.8 ms**，到不了 1.5。

## 3. T6（prefetch / S4）处置

**本 sprint 不落代码刀**（避免在已回退的 S2c 状态上再叠不确定 ±）。

依据：

- T4 后 `aic_mte2≈0.95 ms` 仍高，但主因仍是 **MCH 回灌 Nd2Nz**，不是 Score 侧可 soft-prefetch 的尾部
- S4（减 Dual-AIV barrier / Store∥下一拍）板端 id8 嫌疑在，但收益上限见 `TARGET_1P5` §4（±0.3~0.7），填不满 2.6 ms 缺口
- 下一刀若做：单变量、可回退宏；优先 **Store∥下一 Score** 或 **单 AIV 满行 Post** 试验，不要重开 S2c 除非重做 Prep overlap

## 4. 另立项（才能逼近 1.5）

| 方向 | 为何必要 |
|------|----------|
| **融合**（与上下游 op / 多 stage 合核） | 摊掉 CrossCore + 重复 GM 往返 |
| **改调度**（更大 DEPTH + 真流水、或减 NC 语义等价切分） | 现 DEPTH=2 与 per-sub CV 税太重；朴素 S2c 已证伪 |
| **减工作量**（更少 Neumann iter / 近似、或更大 fractal 摊 Fixpipe） | 16³×(Score+MCH) 的 Fixpipe 次数是硬下限之一 |
| **Ascend950 L0C→L1**（若目标 Soc 升级） | 才可能实现 TARGET 字面「中间不经 GM」 |

## 5. 开关现状（可回退）

- `USE_SCORE_TILE_MMAD=1`
- `USE_MCH_L0_ACC=1` / `USE_MCH_L0_DUAL=1` / `USE_MCH_S2B_STEAL=0`
- `USE_MCH_L1_RESIDENT=1`
- `USE_S2C_BATCH=0`（代码保留）

## 6. 结论

**1.5 ms 作为单算子牵引目标：本路径未达成，且证据表明在现有 MIX CV + 910B Fixpipe 模型下不可靠承诺。**  
推荐把 1.5 转为 **融合 / 调度 / Soc 能力** 的另立项 OKR；本仓保留 ~4.1 ms 最佳配置与可开关实验路径。
