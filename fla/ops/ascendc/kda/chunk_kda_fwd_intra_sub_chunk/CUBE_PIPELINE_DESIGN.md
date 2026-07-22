# ChunkKdaFwdIntraSubChunk Cube：分核 / CV 流水 / Profiling / 后续刀

> 模型目标 shape：`(B=1,H=32,T=8192,K=128,BT=64)` bf16，tiling key 1（MIX_AIC_1_2）。
> 最新 msprof（P3 后，空闲卡 `ASCEND_DEVICE_ID=1`）：设备 Task Duration **≈8.61 ms**（P6 为 8.96 ms）。
> 冲刺 plan：`/root/.cursor/plans/intra_sub_chunk_5ms_push_a1b2c3d4.plan.md`

---

## 1. 分核模型

| 角色 | 数量 / 核 | 职责 |
|------|-----------|------|
| AIC | 1 / MixBlock | Score MMAD（qg@kneg、kpos@kneg）+ MCH Cube 段（Y=L²、X@Y） |
| AIV0 / AIV1 | 2 / MixBlock | 按 **BC 行半区** 拆：`row ∈ [bc*sid/2, bc*(sid+1)/2)` |
| MixBlock | `usedCoreNum`（模型 ~20） | 按 task=`(b,hv,chunk)` 静态调度 |

Dual-AIV 已覆盖：

| 阶段 | AIV0 | AIV1 | 同步 |
|------|------|------|------|
| PrepareSub | 半行 midpoint/exp/score 写 | 半行 | `readyFlag` → AIC |
| PostSubWriteSolve | 半行 tril/β/写 L·X | 半行 | AIV barrier → `solveReady` |
| PostSubMchAdds | 半行 `X += TMP` | 半行 | — |
| PostSubStore | 半行 aqk/akkd | 半行 | AIV barrier |

AIC 调度（P5）：

```text
WaitReady → ComputeMmad → SetDone → ComputeMchAic
  ComputeMchAic: SetDone(Y=L² 前可与 AIV Write 重叠) → Y@Y → WaitSolveReady → X@Y → …
```

AIV 调度（Phase D）：

```text
prep(0)+ready
for i:
  WaitDone(mmad i)
  WriteSolve(i) + barrier + solveReady
  if i+1: prep(i+1)+ready          // ‖ AIC MCH(i)
  MchAdds(i); Store(i); barrier
```

---

## 2. CV / Workspace

- `cmatWs_[slot, plane∈{AQK,AKK}, BC×BC]`：Cube 写出的原始 mat / Post 用的 L
- `solveWs_[slot, plane∈{X,TMP}, BC×BC]`：前代 X 与 MCH TMP
- `SCORE_QUEUE_DEPTH=2`：prep(i+1) 与 MCH(i) 双缓冲
- CrossCore：`readyFlag` / `doneFlag` / `solveReadyFlag`（Catlass Arch）

---

## 3. Profiling

### P6（`prof_intra_sub_chunk_p6`）→ P3（`prof_intra_sub_chunk_p3`）

| 指标 | P6 中位 | P3 中位 | 解读 |
|------|---------|---------|------|
| Task Duration | 8.96 ms | **8.61 ms** | Select tril 约 **-0.35 ms** |
| `aiv_scalar_ratio` | 0.52 | **0.51** | 几乎不动 → tril 环不是主标量源 |
| `aiv_vec_ratio` | 0.15 | 0.12 | — |
| `aic_mac_ratio` | 0.031 | 0.032 | Cube 仍极短（~0.27 ms） |
| `aic_scalar_ratio` | 0.33 | 0.35 | AIC 仍大量空等 AIV |

历史：scalar Post ~52 → ~16 → P6 **8.96** → P3 **8.61**。

**结论：** P3 收益小；残差 scalar 更可能来自 **HardEvent/CrossCore 等待**、PrepareSub、以及 WriteSolve **对角 I 的 prefix×行**。下一刀优先 **P3b 对角 I / 减 sync**，再视情况 P5b `Mmad_ACC`。

---

## 4. 已落地优化（commit）

| 阶段 | Commit | 要点 |
|------|--------|------|
| A–C | `af4ba42` | Store/tril/MCH 向量化起点 |
| D | `9f3ca60` | `prep(i+1) ‖ MCH(i)` |
| P1+P2+P4 | `af981bf` | Prepare 批量；Brcb β；Store 整块；杀 Get/Set |
| P5 | `6c2be7e` | Y@Y‖Add；双 AIV Add |
| P6 | `88d97ab` | WriteSolve/Store 双 AIV 半行 |

坑：Brcb `*RepStride` 以 32B block 计（BC=16→2）；对角勿 `Adds(1)` 错位；`CopyVectorOut` 第三参须 lvalue；忙卡 host avg 可假回归。

---

## 5. 下一刀（按 msprof）

1. **P3 Select tril**（优先）：缩版 `chunk_kda_fwd::SelectCausalRows`，BC=16 单 col-block，替换 `ApplyTrilScaleBeta` 内 prefix×行。预期打 `aiv_scalar`。
2. **对角 I 向量化**（P3 附属或 P3b）：`WriteSolveInputs` 仍 prefix one-hot×行，可再 Select/预置 I。
3. **P5b `Mmad_ACC`**：对标 `solve_tri`，收口跨核 Add；墙钟增益取决于能否缩短 AIV 临界路径。
4. 不做：改 ABI/BC；score 升 fp32；大改 Catlass L1。

验收：空闲卡 msprof + 全量 `test_npu_chunk_kda_fwd_intra_sub_chunk.py`；勿只看 host avg。
