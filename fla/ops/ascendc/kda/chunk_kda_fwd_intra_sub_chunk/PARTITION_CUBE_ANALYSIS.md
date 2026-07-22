# ChunkKdaFwdIntraSubChunk：分核策略与右矩阵复用分析

> 基于计算流程图（单对角 sub-chunk 流水）。
> 算子对标 Triton `chunk_kda_fwd_kernel_intra_sub_chunk`；公开 layout 为 BNSD。

---

## 1. 依赖关系（分核前提）

| 维度                               | 是否有依赖              | 说明                                                                            |
| ---------------------------------- | ----------------------- | ------------------------------------------------------------------------------- |
| chunk ↔ chunk                     | **无**            | 各写自己的对角块，互不读写                                                      |
| 同 chunk 内 sub-chunk ↔ sub-chunk | **无**（对本 L0） | 只写对角`i·BC` 列块；跨 BC 的 off-diag 属于另一 kernel `inter_solve_fused` |
| forward-sub                        | **仅块内**        | 16×16，不跨 sub-chunk                                                          |
| batch / head                       | **无**            | 独立                                                                            |

**结论：** 无跨 task 数据依赖 ⇒ 既可「一 task 一 sub-chunk」扁平并行，也可「一核一 chunk、核内串行 NC」深融合；选哪种由 **Cube 启动/CV 同步开销** 决定，而不是由正确性依赖决定。

---

## 2. 当前分核策略（AIV scalar）

### 2.1 怎么分

```text
totalTasks = B × HV × NT × NC
blockDim   = min(totalTasks, GetCoreNumAiv())
KERNEL     = AIV_ONLY

每个 task：
  解码 (b, hv, chunk, i_sub)
  完成该对角 sub-chunk 全流程（gate → 标量 GEMM → tril → forward-sub → store）
```

- `NT = ceil(T / BT)`（varlen 时为 `chunk_indices` 长度）
- `NC = BT / BC`，`BC = 16` 固定
- 模型例：`B=1, H=HV=32, T=8192, BT=64` → `NT=128, NC=4` → **16384** 个 task

### 2.2 为何对 scalar 合理

- 每个 sub-chunk 独立 → 并行度最大、负载均衡简单
- 热路径是标量三重循环，本身极慢，**调度开销相对可忽略**
- 实现与调试成本低（当前已交付路径）

### 2.3 代价

- 同一 BT 窗口内 `i=0..NC-1` **各自冷启动**，不复用同 chunk 的地址局部性
- 上 Cube 后：每次 MMAD + CV flag 的固定开销会被 task 数放大，扁平分核变劣势

---

## 3. 建议分核策略（Cube 深融合）

### 3.1 怎么分

```text
taskC    = B × HV × NT          # 外层不再含 NC
blockDim = GetCoreNumAic()
KERNEL   = MIX_AIC_1_2

每个 AIC 核绑定一个 (b, hv, chunk)：
  for i_sub in 0 .. NC-1:
      slot = i_sub % DEPTH        # DEPTH=2
      AIV:  prep → scratch[slot] → SetFlag ready
      AIC:  Wait → MMAD×2 → C[slot] → SetFlag done
      AIV:  Wait → tril/scale/β → forward-sub → store
      # 可选：post(i) 与 prep(i+1) 重叠
```

- 模型例：外层 **4096** 个 chunk-task；每核内做 `NC=4` 次 CV 流水

### 3.2 Workspace（示意）

```text
每 AIC 核：
  score[DEPTH][3][BC][K] × sizeof(T)   # QG / W(kpos) / KG(kneg)
  cmat [DEPTH][2][BC][BC] × sizeof(fp32) # Aqk_raw / Akk_raw
DEPTH = 2
```

空 sub（`i_ti >= T`）：仍走成对 ready/done（写零 scratch 或跳过 MMAD 但 flag 平衡）。

### 3.3 对比表

| 项                    | 当前（scalar）      | 建议（Cube 深融合）                       |
| --------------------- | ------------------- | ----------------------------------------- |
| 外层 task             | `B×HV×NT×NC`   | `B×HV×NT`（NC 核内循环）              |
| blockDim              | `GetCoreNumAiv()` | `GetCoreNumAic()` + MIX_AIC_1_2         |
| 工作单元              | 1 sub-chunk / task  | 1 chunk / AIC 核                          |
| 跨 task 依赖          | 无                  | 无（同）                                  |
| Workspace             | 仅 LibApi           | 每核 DEPTH=2 scratch                      |
| CV 同步               | 无                  | 每 sub：AIV ready → AIC MMAD → AIV done |
| 模型 shape 外层任务数 | ~16384              | ~4096                                     |

### 3.4 为何建议深融合

1. **BC=16 时 Cube/CV 固定开销相对大**，把 NC 融进核内可摊薄 flag 与启动成本
2. 同 chunk 内 token 地址连续，MTE2 更友好
3. `DEPTH=2` 可做「AIC 算 sub i ‖ AIV prep sub i+1」，对标 `chunk_kda_fwd` score ping-pong
4. GPU 用 `(NT, NC, B·HV)` 是因为 warp 便宜；NPU 更适合 **少外层任务 + 核内流水**

> 无依赖 ≠ 必须「一 program 一 sub-chunk」。正确性允许扁平；性能上 Cube 路径应深融合。

---

## 4. 右矩阵复用（kneg）

### 4.1 数学

```text
Aqk_raw = qg  @ knegᵀ
Akk_raw = kpos @ knegᵀ
```

两次 MMAD 的 **右矩阵 B 都是 `kneg`**（scratch 中的 KG plane，ColumnMajor 视图）；只换左矩阵 A：`qg` → `kpos`。

### 4.2 复用层级（已锁定为最优解）

| 层级 | 做法 | 状态 |
|------|------|------|
| **L0** | GM 只存一份 KG；两次 MMAD 同指 `blockKNeg` | **必做** |
| **L1/L0 驻留** | 第一次 MMAD 后 **B 留在 L1/L0**；第二次 **只换 A（kpos）与 C（Akk）**，禁止再从 GM 搬 KG | **必做（硬性）** |
| L2 定制双 A 单 B kernel | 仅当 Catlass 无法保留 B 时的后备 | 尽量不走 |

### 4.3 与参考实现的差距

`chunk_kda_fwd` 对同一 `blockKNeg` 连打两次 `blockMmad`，中间 `PipeBarrier` 后 **B 可能被重新搬入**。  
本算子最优解 **强于参考**：显式「B 装一次」；验收看第二次 MMAD 区间无与 KG 等大的 B 侧 MTE2。

### 4.4 Prep 侧含义

AIV prep 仍写三平面：`QG / W(kpos) / KG(kneg)`；**KG 写一次、AIC 读两次（第二次命中 L1）**。

---

## 5. 小结

1. **当前分核（scalar）**：`B×HV×NT×NC` 扁平 — 仅作 key0 fallback。  
2. **最优分核（Cube）**：外层 `B×HV×NT`，核内 NC + DEPTH=2 CV 流水 — **已锁定**。  
3. **右矩阵**：`kneg` GM 一份 + **第一次 MMAD 后 B 留 L1/L0，第二次只换 A/C** — **已锁定为硬性**。  
4. Cube 只替换两次 score GEMM；gate / tril / forward-sub 仍在 AIV。  
5. 执行计划：见 `/root/.cursor/plans/intra_sub_chunk_cube_catlass_7a3f2c1b.plan.md`。

---

## 6. msprof 模型 case（2026-07-21）

### 6.1 采集

```bash
# conda fzy_atk + source set_env.sh
cd flash-linear-attention-npu
msprof --output=prof_intra_sub_chunk_model_pipe --aic-metrics=PipeUtilization \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py
```

- Case：`B=1,H=32,T=8192,K=128,BT=64,bf16`，warmup 3 + iters 5，host avg ≈ **53 ms**
- 文档：[msprof 采集通用命令](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/Profiling/atlasprofiling_16_0010.html)
- **本环境默认导出无 `kernel_detail.csv`**；等价看各 pipe 占比用  
  `mindstudio_profiler_output/op_summary_*.csv`（`OP Type=ChunkKdaFwdIntraSubChunk`）

### 6.2 `op_summary` 关键占比（代表 task，Duration ≈ 52.3 ms）

| 侧 | 指标 | 值 | 解读 |
|----|------|-----|------|
| Wall | Task Duration | ~52.3 ms | 与 host 计时一致 |
| AIC | `aic_mac_ratio` | **0.001**（~72 µs） | Cube MAC 几乎可忽略 |
| AIC | `aic_mte2_ratio` | 0.005（~254 µs） | GM→L1 总搬入很小 |
| AIC | `aic_mte1_ratio` | 0.002 | L1→L0 很小 |
| AIC | `aic_scalar_ratio` | 0.012 | AIC 标量很少 |
| AIC | `cube_utilization(%)` | ~99 | **有 Cube 时**利用率高，但绝对时间极短 |
| AIV | `aiv_scalar_ratio` | **0.426**（~22 ms） | **主瓶颈** |
| AIV | `aiv_vec_ratio` | 0.09（~4.6 ms） | prep 向量路径有贡献但仍次要 |
| AIV | `aiv_mte2/mte3_ratio` | ~0.04 / ~0.05 | 搬运不主导 |

AIC/AIV 计时都贴着 ~51.7 ms：MIX 下 AIC **大量在等 AIV**（pipe ratio 之和 ≪ 1），墙钟由 AIV 决定。

### 6.3 改进空间判断

1. **优先：砍 AIV scalar（~43%）**  
   `PostSub` 里 tril / forward-sub / `StoreAqkRow`/`StoreAkkdRow` 仍是 `GetValue`/`SetValue` 标量环。向量化 store + forward-sub 是最大收益。
2. **次优先：prep(i+1) ‖ post(i)**（DEPTH=2 已备好，CV 仍严格 per-sub sync）  
   可藏一部分 post，但 post 本身是 scalar 热路径，重叠收益上限受 post 时长约束。
3. **Cube / kneg L1 驻留**  
   聚合 `aic_mte2` 仅 ~0.5%，**不是当前墙钟瓶颈**。第二次 MMAD 是否重搬 B 需更细 timeline 才能证伪；即使重搬，相对 52 ms 也可忽略。
4. **不做优先**：再抠 AIC MAC / 升精度 MMAD。

### 6.4 结论

Cube 路径已把 score GEMM 从热路径拿掉；模型 shape 上算子仍慢，是因为 **AIV post/store 标量**，不是 Cube 利用率。下一轮优化应盯 `PostSub`，而不是 Catlass tile。

根因与改进方案（含与 `chunk_kda_fwd` / `solve_tri` 对比）：见 **`SCALAR_BOTTLENECK_ANALYSIS.md`**。  
执行计划：`/root/.cursor/plans/intra_sub_chunk_post_vectorize_9c2e4a1b.plan.md`。

---

## 参考

- 本算子：`fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/`
- CV/GEMM：`chunk_kda_fwd` → `PrepareScoreFactorsBulk` / `ComputeRawAqkAkkCubeBlock`
- GPU：`flash-linear-attention/fla/ops/kda/chunk_intra.py`（`intra_sub_chunk`）
- 流程图 Canvas：`intra-sub-chunk-compute-flow.canvas.tsx`
- Profiling：`prof_intra_sub_chunk_model_pipe/.../mindstudio_profiler_output/op_summary_*.csv`
- Prof 脚本：`torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py`
