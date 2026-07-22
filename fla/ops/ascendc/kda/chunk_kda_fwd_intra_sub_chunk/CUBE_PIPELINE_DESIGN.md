# ChunkKdaFwdIntraSubChunk：压缩事实 + scalar 深挖

> Shape `(1,32,8192,128)` BT=64 bf16 · 空闲卡 `ASCEND_DEVICE_ID=1` · **以 msprof Task Duration 中位为准**  
> 交互分析 Canvas：`intra-sub-chunk-scalar-deep-dive.canvas.tsx`  
> HEAD 参考：`147bc1e`（P3b）

---

## 0. 锁定事实（勿被 host avg / ratio 带偏）

| 阶段 | Task Dur | aiv_scalar **绝对** | aiv_scalar ratio | aic_mac | aic_scalar |
|------|----------|---------------------|------------------|---------|------------|
| 早期 scalar Post | 52.3 ms | ~22 ms | 0.43 | ~0.07 ms | — |
| phased | 15.82 | **4.06** | 0.26 | 0.27 | 3.08 |
| P6 | 8.96 | **4.46** | 0.52 | 0.27 | 2.87 |
| P3 | 8.61 | **4.25** | 0.51 | 0.27 | 2.88 |
| **P3b（最新）** | **8.32** | **4.03** | **0.50** | **0.27** | **2.90** |

精度：P6/P3/P3b 全量 `test_npu_chunk_kda_fwd_intra_sub_chunk.py` **pass**。  
目标：~**5 ms**（缺口 ≈3.3 ms）。

**关键观察：** phased→P3b 墙钟腰斩，`aiv_scalar` **绝对时间几乎不动（~4 ms）** → ratio 升高是**分母效应**，不是又写回了 Get/Set 环。

---

## 1. 分核 / CV（已锁定）

```text
task = B×HV×NT   (~4096)     NC=BT/BC=4 核内循环
MIX_AIC_1_2 · ~20 MixBlock · Dual-AIV 行半区 · DEPTH=2
AIV: prep(0); for i: WaitDone → WriteSolve+barrier+solveReady
         → prep(i+1)+ready ‖ MCH; Adds; Store; barrier
AIC: WaitReady → MMAD → Done → MCH(WaitSolveReady…)
```

- 不要改回 `B×HV×NT×NC` 扁平 Cube（flag×4，更慢）。
- `aic_mac` 恒为 **0.27 ms**；`cube_utilization~96%` 只说明「有 MAC 时很忙」，绝对窗极短。

---

## 2. 为何 scalar 占比仍 ~50%

1. **分母效应**：砍的是 vec/MTE3/旧标量数学；留下的 ~4 ms 同步税不动 → ratio↑。  
2. **同步预算对得上**：~205 chunk/核 × NC4 ≈ **820 sub/核**  
   - AIV scalar 4.03 ms ⇒ **~4.9 µs/sub**  
   - AIC scalar 2.90 ms ⇒ **~3.5 µs/sub**（等 AIV）  
   - MAC ⇒ **~0.32 µs/sub**  
   每 sub：`WaitDone` / `solveReady` / `ready` / 2×AIV barrier / 多对 HardEvent —— 量级覆盖残差。  
3. **Pipe 加总不满 1**：AIV≈0.83、AIC≈0.78 → 另有 17–22% 气泡，部分也进 scalar。  
4. 源码：`SetFlag/WaitFlag` ~60 对、`PipeBarrier` ~50、CrossCore 十余处；`GetPhyAddr` 仅 Select mask 填充。

**结论：** 当前「高 scalar」= **CV/HardEvent 空等 + 控制流**，不是 Post tril/I 数学。

---

## 3. 剩余刀：见效预判

| 方向 | 预期墙钟 | 判据 |
|------|----------|------|
| **合并/删除 HardEvent·CrossCore** | **−0.5~−1.5 ms**（最值得） | 绝对 scalar 钉在 sync 量级 |
| Prep 减负（mid 上提、少 Zero、tile 事件合并） | −0.3~−1.0 ms | Prep 仍在 AIV 临界路径 |
| Post 改非对称（单 AIV Post / 去 barrier） | ±0.5（可能负） | BC=16 半行算力可能 < barrier |
| **P5b Mmad_ACC** | −0.2~−0.8，**仅当能删一次 solveReady 往返** | MAC 只有 0.27 ms |
| 再抠 Select/tril/I | <−0.2 | P3/P3b 已验证边际 |
| 外层改回 ×NC | 变慢 | 已否决 |

到 5 ms 必须动 **握手次数或 Prep**，单靠 P5b/再向量化不够。

---

## 4. 分核还能挖什么

| 已做对 | 可试 | 不要 |
|--------|------|------|
| `B×HV×NT` + 核内 NC | 同 chunk **合并 solveReady** / 批 MMAD 再 MCH（加 WS、减 flag） | 扁平 ×NC |
| DEPTH=2 prep‖MCH | **非对称 AIV**（Post 单核、另一核只 prep） | 为并行而并行半行若 barrier 更贵 |
| Dual-AIV 行拆 | 一 task 打包 2 chunk（摊前奏，通常小） | 改 ABI/BC |

---

## 5. 验收口径

- 看 **`Task Duration` + `aiv_scalar_time(us)` 绝对值**，少看 ratio。  
- 空闲卡；全量精度不回退。  
- 下一步优先：**HardEvent/CrossCore 审计** → Prep 减负 → 再考虑非对称 AIV / 有条件的 P5b。

---

## 6. 已合入（摘要）

`af4ba42` A–C → `9f3ca60` D → `af981bf` P1/2/4 → `6c2be7e` P5 → `88d97ab` P6 → `74107d1` P3 → `147bc1e` P3b  

旧文：`PARTITION_CUBE_ANALYSIS.md`（分核锁定）、`SCALAR_BOTTLENECK_ANALYSIS.md`（~52 ms 时代，历史参考）。
