# ChunkKdaFwdPrepare（A2/A3）Stage 划分（最终稿 · 培训文档同格式）

> 格式对齐《线性 Attention 算子设计培训：Stage 划分与 4-Head 调度》§3.1/§3.2/§4。
> 公式以 ATK 的 FP64 独立标杆 `tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py`
> 的 `_s4_scores` / `_block_inverse` / `_reference` 为准。
> 配套工程细节（数据路径表、调度与 flag 配平、空间预算、验收判据）见 `design_a2a3_stage_final.md`；
> 空间逐项字节账见 `design_a2a3_budget_4head_vs_2head.md`。

> **术语（arch22 = A2/A3）**：A5/arch35 的 AIV 用 **RegBase VF**（`__simd_vf__ inline` + `AscendC::MicroAPI`
> 的 RegTensor/MaskReg/LoadAlign/StoreAlign），一个 Stage = 一次 VF 调用；**A2/A3 没有 VF/RegBase**，
> 用普通 AscendC 向量指令（`LocalTensor` + `Add/Mul/Cast/Exp/...`）。代码里的 `V0Vf/V1Vf/V3Vf/V6Vf`
> 只是**与 A5 同名的向量函数**（工程结构规范 §4.4 要求两平台 Stage 函数同名同序），**不代表 VF 语义**。
> 因此约束 12 在 A2/A3 的等价表述是：**一个 Stage 的数据一次搬完、一轮向量运算算完，不分趟搬运、不分 pass**；
> 容量约束来自 **UB（184 KiB）与 72 KiB 私有区**，不是 VF 寄存器。Stage 边界仍与 A5 一致，是因为两平台必须同名同序。

## 3. 数学语义

### 3.1 公式

输入：
q, k, v, g, beta, scale, chunk_size(=64), epsilon, use_qk_l2norm_in_kernel,
use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, safe_gate,
lower_bound, use_exp2, a_log, dt_bias, cu_seqlens, chunk_indices, backward_mode

输出（13 槽位）：
gk, aqk, akk, w, u, qg, kg, qg_scaled, q_hat, k_hat, q_rstd, k_rstd, beta_eff

记号：单个 chunk 行数 `M ≤ 64`（实际 `validRows`），`K = V = 128`；子块 `S = 4` 个 16 行 band；
`E(x) = 2^x`（`use_exp2`）或 `exp(x)`；`Glast = G[末行]`。

```text
# V0：归一化 + gate + beta
q_rstd = rsqrt(Σ_d q² + ε) ;  q_hat = bf16(q · q_rstd)      # 关闭时 q_hat=q, q_rstd=1
k_rstd = rsqrt(Σ_d k² + ε) ;  k_hat = bf16(k · k_rstd)

Δg[i,d] = g − exp(a_log)·softplus(g + dt_bias)                  # GateMode = Softplus
        | lower_bound / (1 + exp(−exp(a_log)·(g + dt_bias)))     # GateMode = SafeSigmoid
        | g                                                      # GateMode = PrecomputedStep
G = cumsum_i(Δg) · (1/ln2 if use_exp2 else 1)                    # chunk 内累计，保存于 log2/自然域
beta_eff = beta | sigmoid(beta) | 2·sigmoid(beta)                # 三模式

# V1：S=4 分带（每 16 行 band 取中点行作参考）
Gref[s] = G[(band_begin + band_end) // 2]
Qplus[i]  = bf16(q_hat[i] · clampE(G[i] − Gref[s(i)]))           # clamp 到 base-2 [-126, 120]
Kplus[i]  = bf16(k_hat[i] · clampE(G[i] − Gref[s(i)]))
Kminus[s] = bf16(k_hat[j] · clampE(Gref[s] − G[j])), j ∈ [0, band_end)   # 前缀 16/32/48/64 行

# C2：4 个 stacked band MMAD（每个 band 同时出两个数学乘积）
rawAqk_s = Qplus[band_s] @ Kminus[s]ᵀ        # [16, N_s]
rawAkk_s = Kplus[band_s] @ Kminus[s]ᵀ        # [16, N_s]      N_0..3 = 16/32/48/64

# V3：因果 mask + beta + 叶子逆 + 跨叶修正
Aqk[row, :row+1] = rawAqk[row, :row+1] · scale
L[row, :row]     = rawAkk[row, :row] · beta_eff[row]              # 严格下三角
Akk = (I + L)⁻¹  （按 2×32 叶块求解）
      X0 = (I + L₀)⁻¹                                             # 叶 0：行 0..31
      X1 = (I + L₁)⁻¹                                             # 叶 1：行 32..63
      B  = L₂₁                                                    # L 的左下 32×32 块
      Akk[32:64, 0:32] = −X1 @ (B @ X0)                           # M ≤ 32 时无 B/X1/negX1

# C4/C5：跨叶修正（仅 M > 32）
T   = B  @ X0
q10 = (−X1) @ T

# V6：post-WU 侧的派生量
qg         = bf16(q_hat · clampE(G))                              # clamp base-2 [-80, 80]
qg_scaled  = bf16(fp32(qg) · scale)                               # 消费已 BF16 舍入的 qg
kg         = bf16(k_hat · clampE(Glast − G))
K_positive = bf16(k_hat · clampE(G))                              # 与 qg 同一指数因子
K_beta_g   = bf16(fp32(K_positive) · beta_eff)
V_beta     = bf16(fp32(v) · beta_eff)

# C7：WY 表示
W = Akk @ K_beta_g        # [M, 128]
U = Akk @ V_beta          # [M, 128]
```

### 3.2 依赖关系

```text
q_hat/k_hat ─┬─► Qplus/Kplus/Kminus ─► (C2) rawAqk ─► Aqk
             │                          (C2) rawAkk ─► L ─► Akk
             │                                            │
             │                          B/X0/X1/negX1 ─► T=q10 ─► Akk[32:64,0:32]
             │                                            │
             └────────────────────────► K_beta_g ────────┴─► (C7) W
                                        V_beta ────────────┴─► (C7) U

G（gate 累计）─┬─► Gref ─► Qplus/Kplus/Kminus（同上）
               ├─► qg / kg / K_positive
               └─► （V6 只依赖 V0 的 q_hat/k_hat/G/beta_eff 与输入 v）
```

关键结论：**C2 → V3 → C4/C5 → C7 是一条硬串行链**（每步都消费上一步的矩阵结果）；
而 **V6 只依赖 V0 的产物（`q_hat/k_hat/G/beta_eff`）与输入 `v`，不依赖 C2/V3/C4/C5 的任何结果**——
这一点决定了后文的调度差异。

## 4. Stage 划分

| Stage | 执行单元 | 计算 |
| ----- | -------- | ---- |
| V0 | Vector | `q_rstd/k_rstd`、`q_hat/k_hat`；`Δg`（三种 gate 模式）+ `cumsum` → `G`；`beta_eff`（三模式） |
| V1 | Vector | `S=4` 分带参考 `Gref`；`Qplus = q_hat·E(G−Gref)`、`Kplus = k_hat·E(G−Gref)`、`Kminus[s] = k_hat·E(Gref−G)`（前缀 16/32/48/64） |
| C2 | Cube | 4 个 stacked band MMAD：`rawAqk_s = Qplus_s @ Kminus_sᵀ`、`rawAkk_s = Kplus_s @ Kminus_sᵀ`（N=16/32/48/64，M=32，K=128） |
| V3 | Vector | `Aqk = tril(rawAqk)·scale`；`L = tril(rawAkk, −1)·beta_eff`；两个 32×32 叶子逆 + 跨叶修正 → `Akk`（含 `B/X0/X1/negX1`） |
| C4 | Cube | `T = B @ X0`（仅 `M > 32`） |
| C5 | Cube | `q10 = (−X1) @ T`（仅 `M > 32`） |
| V6 | Vector | `qg`、`qg_scaled`、`kg`、`K_positive`、`K_beta_g`、`V_beta` |
| C7 | Cube | `W = Akk @ K_beta_g`、`U = Akk @ V_beta` |

这样划分有四个原因：

1. **执行单元唯一**：8 个 stage 严格二分（4 个 Vector + 4 个 Cube），满足"每个 stage 只能有矩阵乘
   或向量操作之一"。
2. **Cube 不依赖本 stage 新产物**：`C2` 只读 V1 已落 GM 的 `Qplus/Kplus`；`C4` 读 V3 的 `B/X0`；
   `C7` 读 V3 的 `Akk` 与 V6 的 `K_beta_g/V_beta`。因此每个 Cube stage 的输入都在进入时已就绪。
3. **叶子逆必须在 Vector**：`Akk=(I+L)⁻¹` 的分块求解（`X0/X1` 三角求逆 + 跨叶修正）是逐元素/逐块
   的向量运算，且 `L` 由 `rawAkk` 在同一个 Vector stage 内做 mask+beta 得到，可在**同一个 Stage 内一轮算完**
   （A5 是一次 VF；A2/A3 无 VF，用普通向量指令实现同一 Stage）。
4. **V0/V1 为什么不合并**（注意：这不是 A2/A3 的容量约束）：在 A2/A3 上，V0 与 V1 本来就复用
   **同一段 72 KiB 私有区**（`Q/K` 各 16 KiB + `gate/Kminus` 40 KiB，其余状态槽在 `0x10000+`），
   所以"合并"的内存代价与现在**相同**。真正不合并的原因是：
   - 划分依据来自 **A5 侧的一次 VF 约束**（一个 Stage 的数学必须在一次 RegBase VF 内完成）；
   - 工程结构规范 §4.4 要求两平台 **Stage 函数同名同序**，所以 A2/A3 跟随 A5 的边界；
   - 约束 15 对"连续 vector"的担忧已被"V0/V1 在同一握手回合内、跨核成本为 0"消掉。
   ⇒ **A2/A3 的 Stage 边界由 A5 的 VF 约束 + 跨平台一致性决定，而不是由 A2/A3 自身的 UB 预算决定**。
   若将来只为 A2/A3 做更激进的划分（合并 V0/V1、拆分 V3 等），必须在 `design.md` 写明平台差异表，
   并证明不破坏逐 Stage 验证（§7.2）。

> 同理 `C4`/`C5` 也不能合并成一个 Cube stage：`C5` 依赖 `C4` 的 `T`，合并会让 Cube stage 依赖
> 本 stage 自身的新产物。

## 5. 与原来（现行代码）Stage 划分的差异

### 5.1 原来是什么样

| 项 | 现行实现（`op_kernel/arch22/*` + `docs/design.md §3`） |
|---|---|
| Stage 边界 | 8 个：`V0 → V1 → C2 → V3 → C4 → C5 → V6 → C7` |
| 执行单元 | V0/V1/V3/V6 = Vector；C2/C4/C5/C7 = Cube（共 8 次 `Mmad` 调用） |
| 跨核回合 | **3 个** ready/free 回合：<br>R1 = `V0+V1` ⇄ `C2`；R2 = `V3` ⇄ `C4`（`C5` 无跨核等待）；R3 = `V6` ⇄ `C7` |
| 两个 AIV 分工 | pair wave：pair0={h0,h1}、pair1={h2,h3}；`localHead = pair*2 + aiv`，即 AIV0 做 h0/h2、AIV1 做 h1/h3 |
| 尾部调度 | A2/A3 **不启用** 4-head 尾部展开（仅 A5 dense 在"chunk 数不整除核数"时启用） |
| 资源结论的表述 | 未把"调度窗口 / 驻留份数 / buffer 槽数"三层"4"分开写清 |

### 5.2 最终稿是什么样（逐项对比）

| # | 项 | 原来 | **最终** | 差异依据 |
|---|---|---|---|---|
| 1 | **Stage 边界** | 8 个（V0/V1/C2/V3/C4/C5/V6/C7） | **8 个，完全相同（同名同序）** | 约束 1/2/12 —— 见 §4 的四条原因；本轮分析没有找到需要拆分或合并边界的证据 |
| 2 | **跨核回合数** | **3** | **2** | V6 只依赖 V0（§3.2），可提前；C4/C5/C7 都在 V3 之后且彼此只需 L1 内部事件（C5 原本就无跨核等待） |
| 3 | **V6 的位置** | `C5` 之后（独立回合 3） | **`V1` 之后**（与 `C2` 重叠） | 关键路径从 `V0V1+C2+V3+C4C5+V6+C7` 变为 `V0V1+max(V6,C2)+V3+C4C5+C7` |
| 4 | **尾部 4-head 展开** | A2/A3 走原调度（尾波比 A5 差） | **移植 A5 的尾部展开** | 培训 §7.2 同款手法实测 +1.10% / +4.55%；字节账确认只改任务映射、不改每 head 工作集 |
| 5 | **资源结论** | 未分层表述 | **4-head 调度窗口 + 2-head UB 驻留 + 4 份 L1 head lane / workspace slot** | UB 184 KiB = 2×72 + 40（正好用满），4-head 驻留需 328 KiB ⇒ 不可能；L1 368/512、workspace 4 slot ⇒ 4-head 窗口可行 |
| 6 | （附带）**C7 RHS 预取** | 无 | C4/C5 期间把 `K_beta_g/V_beta` 从 GM 预取到 L1 | L1 余量 144 KiB |

### 5.3 一句话说明"我有什么不同"

**Stage 划分本身不变**（8 个 stage、边界与执行单元归属都与现行实现一致）——
因为约束 1（执行单元唯一）、约束 2（Cube 不依赖本 stage 新产物）、约束 12（一次搬运 + 一轮算完、不分 pass）
已经把边界唯一确定了（A5 侧由"一次 VF"与上面三条共同锁定，A2/A3 按 §4.4 同名同序跟随），
重新推导没有产生新的切法。

**真正的差异在"调度归属"和"资源账"**：

1. `V6` 从第 3 个回合提到第 1 个回合（藏在 `C2` 窗口里）⇒ **跨核回合 3 → 2**；
2. `C4→C5→C7` 合并进第 2 个回合；
3. 把 A5 已经实现、A2/A3 还没走的**尾部 4-head 展开**补上；
4. 把"4"的三种含义写清楚：**调度窗口 4 / L1 lane 4 / workspace slot 4，但 UB 驻留只能 2**。

### 5.4 差异 #2/#3 的落地前置条件（不满足则退回原来）

必须同时成立（详见 `design_a2a3_stage_final.md §4.2`）：

1. V6 的 GM 输出（`K_beta_g/V_beta`）对 AIC 可见（`DataSyncBarrier<DDR>`/FIX 事件覆盖）；
2. V6 与 V3 在同一 AIV 上**顺序执行**（共用 72 KiB 私有区，原地换义）；
3. 共享区换义顺序正确（`kV3BetaEff = kSharedBase` 在 V6 使用共享区之后）；
4. 回合数变化后 flag 严格配平（A2/A3 为 mode 0x2 集合同步：ready 6→4、free 8→6），并跑 ≥3 轮 hang 看护。

另需先补 4 个实测值（A2 真机、同机同频、3 warmup×10 取中位）：`V6` 时长、`C2` 时长、`δ_flag`、尾波比。
决策规则：`min(V6,C2) ≥ 总时长 5%` ⇒ 采用差异 #2/#3；否则只做 #4/#5/#6。

## 6. 什么情况下才需要动这 8 个 stage 边界

只有当下列条件之一成立时，才需要重新划边界（当前都不成立）：

1. 某个 stage 的活跃工作集超过 UB 可用量（A2/A3 为 184 KiB，约束 8）——例如 V3 的 compact raw
   与 Akk 打包同时膨胀；
2. 算法本身变化导致某条依赖消失或新增（例如 DPLR/bg 路径进入本算子的模板轴）；
3. 平台能力变化（A2/A3 出现 L0C→UB 直连，C2 的 GM relay 可以取消，`C` 的落点随之改变）。

## 7. 按 PR#833 工作流，本 Stage 划分还要补什么

PR#833（`docs/算子自动开发工作流.md`）**不提供 stage 划分方法**，但它给出两条对本任务直接生效的规则：

### 7.1 进入阶段：本次属 CANNBot 03，不是 05

| 开发场景 | 进入位置 |
| --- | --- |
| **Stage、tiling、同步或资源方案修改** | **CANNBot 03** |
| 涉及设计变化的性能优化 | CANNBot 03 |
| 不改变设计的性能优化 | CANNBot 05 |

⇒ 本任务（改 Stage 归属 / 调度回合 / 资源账）**必须走 03 → 04**；其中：

- **D1（V6 提前）/ D2（回合 3→2）**：改了同步协议与资源生命周期 ⇒ **属设计变化，走 03**；
- **D3（尾部 4-head 展开）**：改任务映射/tiling ⇒ **走 03**；
- **D4（C7 RHS 预取）**：不改设计假设 ⇒ 可在 04/05 内做（但需回归保证）。

CANNBot 03 的产出物固定为 **`docs/design.md` 里的 Stage / 资源 / 同步 / tiling / 性能方案**——
本目录下的 5 份分析文档需要**收编进 `docs/design.md` 的对应章节**，而不是散在旁路文档里。

### 7.2 CANNBot 04 的"逐 Stage 定向验证"是我们目前缺的能力

PR#833 对 04 的要求原文要点：

```text
host main() 负责 ACL 初始化、tiling、device 内存和 workspace，直接执行 kernel<<<blockDim,...,stream>>>，
拷回当前 Stage 或整 kernel 输出，再使用同一 reference.py 比较。
每轮只运行失败用例、受影响 Stage 和最小边界集合：
  1) 当前 Stage 实现后，运行从 Stage 0 到当前 Stage 的最小合法用例并比较中间输出；
  2) 修复或优化只回归受影响 Stage、原失败 case、目标模型 shape 和必要边界；
  3) 全部 Stage 拼接后跑最小整 kernel 冒烟；
  4) 每轮性能候选先过定向精度，只有最终候选进 full 验收。
```

现状核对（chunk_kda_fwd_prepare）：

| 要求 | 现状 | 缺口 |
|---|---|---|
| 直调工程（host main + 直调 kernel + 拷回中间输出） | **无 `examples/`**；只有 ATK 端到端 executor | ❌ 需新建 |
| `tests/atk/<op>/reference.py` 作为唯一标杆 | 已有（FP64 独立标杆，在 executor 内） | 需迁出为 `reference.py` 并暴露**中间量**接口 |
| 每个 Stage 的中间量可 dump/比对 | 无规格 | ❌ 需补规格表 |
| 从 Stage 0 到当前 Stage 的最小可运行用例 | 无定义 | ❌ 需补 |

### 7.3 由此得到 3 条对划分本身的约束/可优化项

1. **原址换义必须按 Stage 记录**：`C2` 的 raw score 与 `C4` 的 `T` 共用 GM relay 前 4 KiB；
   `V3` 与 `V6` 共用 72 KiB 私有区（`kV3*` 与 `kV6*` 同基址）。逐 Stage dump 时同一地址在不同
   Stage 语义不同 ⇒ 设计里必须写清"地址 → 哪个 Stage → 什么语义"，否则验证会歧义。
2. **并行调度下 dump 要带"代数"**：D1 把 V6 提前后，`V6` 与 `C2` 并行；从 Stage 0 到当前 Stage
   的"最小合法用例"必须能区分**依赖顺序**与**调度顺序**（flag 已保证入口就绪，但 dump 需标明
   该 Stage 消费的是哪一代输入）。
3. **把 Stage 表扩成四列**（这也是 03 产出物的要求）：

   | Stage | 执行单元 | 计算 | **中间量（名 → 落点 → 形状/dtype → reference 对应量）** |
   |---|---|---|---|

   —— 这一列是从"能实现"走向"能逐 Stage 验证"的必要信息，也是目前 `docs/design.md` 八阶段表缺的内容。

### 7.4 对"stage 有没有可以优化"的结论

- **边界：PR#833 没有提供新的划分方法**，按约束 1/2/12 推导的 8 个 stage 仍然是最小且唯一的切法；
- **可优化的是"可验证性"与"调度"**：先补齐 7.2 的三件缺口（直调工程 / 中间量规格 / Stage 0..k 最小用例），
  再按 03 → 04 落地 D1–D3；D4 可在 04 内直接做。
- **唯一的"边界级"可探索项（P2-3）**：因为 **A2/A3 无 VF**，其 Stage 容量上限只由 UB 决定（不是 VF 寄存器），
  所以原则上可以**只为 A2/A3** 重切边界（例如把 `V3` 拆成 `mask+beta` 与 `叶子逆+跨叶修正` 两段）。
  准入条件：分 Stage 计时证明 V3 是瓶颈 + UB 账证明拆完不挤占其它常驻 + 在 `design.md` 写明平台差异表，
  并证明逐 Stage 验证不被破坏。当前证据不支持先做，列为备选。
