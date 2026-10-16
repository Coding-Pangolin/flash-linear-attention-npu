# ChunkKdaFwdPrepare（A2/A3）最终 Stage 划分

> 状态：**划分定稿**（8 个 stage、落点、同步协议、空间预算）；**调度优化待实测确认**（§6）。
> 依据：培训文档《Stage 划分与 4-Head 调度》19 条约束 / §6 调度 / §7 KernelA 实测；
> 本仓 `docs/design.md`（八阶段流水、§5 分核、§6 静态内存、§7 同步）；
> `design_a2a3_stage_redesign.md`、`design_a2a3_budget_4head_vs_2head.md`、`design_a2a3_opt_opportunities.md`。
> 代码基线：`op_kernel/chunk_kda_fwd_prepare_policy.h` + `op_kernel/arch22/{cube,vec}.h`。

## 1. 速览

| 项 | 结论 |
|---|---|
| Stage 数 | **8 个**：`V0 → V1 → C2 → V3 → C4 → C5 → V6 → C7`（4 个 vector + 4 个 cube） |
| 执行单元 | AIV：V0/V1/V3/V6（A5 = 各一次 RegBase VF；**A2/A3 无 VF，用普通向量指令，代码里是与 A5 同名的 `V0Vf/V1Vf/V3Vf/V6Vf` 函数**）；AIC：C2/C4/C5/C7（共 8 次 `Mmad` 调用） |
| 调度窗口 | **4 head**（`kHeadsPerGroup=4`）；任务 = `chunk × 4head` |
| UB 驻留 | **2 head**（每 AIV 一份私有工作集 72 KiB ×2 + 共享 40 KiB = **184 KiB，正好用满**） |
| L1 | 4 head lane 288 KiB + X0/NegX1/T/Akk 80 KiB = **368/512 KiB**（余 144 KiB） |
| 跨核回合 | 现状 **3 个** ready/free 回合；建议 **2 个**（V6 提前 + C4/C5/C7 合回合，待实测确认） |
| 与约束的偏离 | **1 处平台被迫**：C2 的 raw score 只能经 GM relay（A2/A3 无 L0C→UB） |

## 2. 最终 Stage 表

| Stage | 单元 | 计算 | 输入 | 输出 / 落点 |
|---|---|---|---|---|
| **V0** | AIV | Q/K L2 norm（`rstd=rsqrt(Σq²+eps)`、`q_hat=bf16(q·rstd)`）；beta 三模式；gate 三模式（softplus / safe-sigmoid / 预计算 step）+ **chunk 内 cumsum**；产出 `gk` | GM: `q,k,gate,beta,(a_log,dt_bias)` | UB 私有区（临时）；GM: `q_hat/k_hat` 各 16 KiB、`gk` 32 KiB、`beta_eff` 256 B、`q_rstd/k_rstd` 各 256 B（按输出模式） |
| **V1** | AIV | 16 行分带（S=4）取中点参考行 `Gref`；`Qplus/Kplus/Kminus` | UB 私有区 + **共享 G 32 KiB** | **GM payload 72 KiB**（Qplus 16 + Kplus 16 + Kminus 40）——约束 6：vector→cube 必须经 GM |
| **C2** | AIC | 4 个 stacked band MMAD（N=16/32/48/64，M=32，K=128），一次覆盖 8 个数学乘积：raw Aqk（下三角带）/ raw Akk | **L1 head lane 288 KiB** ← GM payload | **GM relay 20 KiB**（约束 4 的唯一妥协） |
| **V3** | AIV | causal mask + beta 缩放；两个 **32×32 叶子逆**（right-looking 分块前代）；跨叶修正 → `Aqk/B/X0/X1/negX1/Akk` | GM relay 20 KiB + `beta_eff` | GM payload（Aqk/Akk 与修正块） |
| **C4** | AIC | `T = B@X0`（32³，**仅 M>32**） | L1: B / X0（每 head 4 KiB 槽位）← GM payload | **L1: T 4 KiB/head** |
| **C5** | AIC | `q10 = negX1@T`（32³，**仅 M>32**） | L1: T + negX1 | **L1: q10** |
| **V6** | AIV | `qg=bf16(q_hat·E(G))`、`qg_scaled=qg·scale`、`kg=bf16(k_hat·E(Glast−G))`、`K_positive=bf16(k_hat·E(G))`、`K_beta_g=K_positive·beta_eff`、`V_beta=v·beta_eff` | GM: `q_hat/k_hat`(V0 产物)、`gk`(V0 产物)、`v`；UB: `beta_eff` | **GM**: qg16 / qg_scaled16 / kg16 / K_beta_g16 / V_beta16（每 head） |
| **C7** | AIC | `W = Akk@K_beta_g`、`U = Akk@V_beta`（m×128×m，m=32/64） | GM: `Akk`(V3) / `K_beta_g`,`V_beta`(V6) | 输出 `w/u`（每 head 16 KiB） |

**Stage 边界的三条硬理由**（不允许再合并）：

1. 约束 1：每个 stage 只能有矩阵乘或向量操作之一 —— 上表 8 个 stage 已严格二分；
2. 约束 2：cube stage 不依赖本 stage 新产物（C2 只读 V1 的 Qplus/Kplus；C4 读 V3；C7 读 V3+V6）；
3. 约束 12：一个 stage 一次搬完 + **一轮算完、不分趟搬运** —— **V0 与 V1 不合并**。
   ⚠ 理由**不是** A2/A3 容量不够：A2/A3 上 V0/V1 复用同一段 72 KiB 私有区（`Q/K` 32 KiB + `gate/Kminus` 40 KiB），
   合并的内存代价与现在相同。真正原因是 **A5 侧的一次 VF 约束** + §4.4 要求两平台 Stage 函数**同名同序**
   （否则无法逐 Stage 对比与验证，见 §7.2）；约束 15 的担忧已由"V0/V1 在同一握手回合内、跨核成本为 0"消解。

> **术语**：A5/arch35 的 AIV 用 RegBase VF（`__simd_vf__` + MicroAPI）；**A2/A3 没有 VF/RegBase**，
> 用普通 AscendC 向量指令，代码里的 `V0Vf/V1Vf/V3Vf/V6Vf` 只是与 A5 同名的向量函数（§4.4 要求两平台同名同序）。
> 约束 12 在 A2/A3 的等价表述是"一次搬完 + 一轮向量运算算完、不分 pass"，容量约束来自 UB 而不是 VF 寄存器。

## 3. 数据路径（谁放哪儿、谁消费）

```text
GM 输入 ──► V0(UB 私有)  ──► q_hat/k_hat(16+16K) ──► GM(workspace context)
                          ──► gk(32K)             ──► GM(context) ──► V1/V3/V6
                          ──► beta_eff(256B)      ──► GM(context) ──► V3/V6/C7

GM gk ──► V1(UB 私有 + 共享 G 32K) ──► Qplus/Kplus/Kminus(72K) ──► GM(payload)
GM(payload) ──► C2 ──► L1 4 head lane(288K)   [Qplus/Kplus/Kminus 进 L1]
C2 C(20K) ──► GM relay ──► V3(UB 私有) ──► Aqk/Akk + B/X0/X1/negX1 ──► GM(payload)
GM(payload) ──► C4 ──► L1 T(4K/head) ──► C5 ──► L1 q10        [仅 M>32]
GM(q_hat/k_hat/gk/v) ──► V6(UB 私有) ──► qg/qg_scaled/kg/K_beta_g/V_beta ──► GM
GM(Akk/K_beta_g/V_beta) ──► C7 ──► 输出 w/u
```

**唯一的平台被迫妥协**：C2 的 raw Aqk/raw Akk 无法 L0C→UB（A2/A3 无该通道），
只能 fixpipe→**GM relay 20 KiB**，由 V3 从 GM 读；C4 再复用该 relay 的前 4 KiB 放 `T`（原址换义，
靠 ready/free 回合保证 WAR）。这条要在 `design.md` 里作为"约束 14 允许的妥协"显式记录。

## 4. 调度与同步协议

### 4.1 现状（3 回合）

```text
R1  AIV: wait free[pair] → V0 → V1        → set ready[pair]      (pair = 0: h0,h1; 1: h2,h3)
R1  AIC: wait ready[pair] → C2(2 head)    → set free[pair]
R2  AIV: wait free[pair] → V3             → set ready[pair]
R2  AIC: wait ready[pair] → C4(2 head)    → set free[pair]  ;  C5（无跨核等待）
R3  AIV: wait free[pair] → V6             → set ready[pair]
R3  AIC: wait ready[pair] → C7            → set free[pair]
```

A2/A3 是 **mode 0x2 集合同步**：AIC 一次 `CrossCoreSetFlag<0x2,…>` 唤醒本 block 两个 AIV；
两个 AIV 都 `set` 之后 AIC 的 `wait` 才会满足。ready = 0/1，free = 2/3。

**flag 配平账（每 4-head 组，现状）**：AIC `set free` = 初始 2 + C2 后 2 + C4 后 2 + C7 后 2 = **8**；
AIV `wait free` = 2×3 + 收尾 drain 2 = **8** ✓；AIV `set ready` = 2×3 = **6**；AIC `wait ready` = **6** ✓。
（不配平的表现是 hang，不是数值错。）

### 4.2 建议（2 回合，待实测确认）

```text
R1  AIV: wait free[pair] → V0 → V1 → V6   → set ready[pair]      ← V6 提前，与 C2 重叠
R1  AIC: wait ready[pair] → C2(2 head)    → set free[pair]
R2  AIV: wait free[pair] → V3             → set ready[pair]
R2  AIC: wait ready[pair] → C4 → C5 → C7  → set free[pair]
```

**flag 配平账（每 4-head 组，建议）**：AIC `set free` = 初始 2 + C2 后 2 + C4/C5/C7 后 2 = **6**；
AIV `wait free` = 2 + 2 + 收尾 drain 2 = **6** ✓；AIV `set ready` = 2 + 2 = **4**；AIC `wait ready` = **4** ✓。

**变更为满足需同时成立 4 条**（任一不成立 ⇒ 退回 3 回合现状）：

1. V6 的 GM 输出（`K_beta_g/V_beta`）对 AIC 的可见性：确认 `DataSyncBarrier<DDR>` / FIX 事件覆盖
   （否则 C7 会"偶发读到旧数据"，属隐性竞态）；
2. V6 与 V3 在**同一 AIV 上顺序执行**（两者共用私有区 72 KiB，原地换义，不能并行）；
3. 共享区换义顺序正确：`kV3BetaEff = kSharedBase` 必须在 V6 使用共享区之后（V6 先、V3 后）；
4. 回合数变化后 flag 严格配平（见上账），并跑 ≥3 轮 hang 看护。

## 5. 空间预算（4-head 是否够、不够就 2-head）

### 5.1 UB（A2/A3）——**只能 2-head 驻留**

硬件 192 KiB；可用 `0x2E000` = **184 KiB**（尾 8 KiB 保留）。布局：私有0(72) | G(32) | scratch(8) | 私有1(72)

| 区域 | 偏移 | 大小 | 用途 |
|---|---|---|---|
| private[0]（AIV0） | 0x00000 | **72 KiB** | V0/V1/V3/V6 的工作集（按 stage 原地换义） |
| 共享 G | 0x12000 | **32 KiB** | FP32 G（V0 产出 → V1/V3/V6 消费） |
| 共享 scratch | 0x1A000 | **8 KiB** | Stage 内向量临时区 + beta gather offsets（A2/A3 无 VF） |
| private[1]（AIV1） | 0x1C000 | **72 KiB** | 同 private[0] |
| 合计 | — | **184 KiB** | **正好用满**（无余量） |

| 方案 | 需求 | 184 KiB 容量 | 结论 |
|---|---|---|---|
| **2-head 驻留**（每 AIV 一个 head 工作集，pair 分两批） | 2×72 + 40 = **184 KiB** | 184 KiB | ✅ **正好，采用** |
| 4-head 驻留（4 份工作集同时在 UB） | 4×72 + 40 = **328 KiB** | 184 KiB | ❌ **缺 144 KiB** |

> 每 head 的 72 KiB 是 V1 产出 `Qplus16+Kplus16+Kminus40` 的硬需求，压不下去；
> 因此 **UB 侧 2-head 驻留是唯一可行解**。这不影响 4-head 调度窗口的收益——培训 §7.2 已证明
> 4-head 的收益来自"任务分配更细、尾波更均衡"，而非"四份输入同时驻留"。

### 5.2 L1 ——**4-head 够，8-head 不够**

| 区域 | 偏移 | 大小 |
|---|---|---|
| 4 × head lane（C2 payload） | 0x00000–0x48000 | **288 KiB** |
| X0（每 head 4 KiB 槽位） | 0x48000 | 16 KiB |
| NegX1 | 0x4C000 | 16 KiB |
| T（每 head 4 KiB quadrant） | 0x50000 | 16 KiB |
| Akk（每 head stride 8 KiB） | 0x54000 | 32 KiB |
| **峰值** | — | **368 KiB / 512 KiB（余 144 KiB）** |

| 窗口 | 需求 | 结论 |
|---|---|---|
| 4-head | 288 + 80 = 368 KiB | ✅ 余 144 KiB（可给 C7 RHS 预取） |
| 8-head | 576 + 80 = 656 KiB | ❌ 超 512 KiB |

### 5.3 workspace（GM）

每 slot = `0x1A400`(105 KiB = 33 KiB context + 72 KiB payload) + arch22 relay `0x5000`(20 KiB) = **125 KiB**；
每 workgroup **4 slot = 500 KiB**（28 核约 14 MB，GM 无压力）。

### 5.4 结论

**4-head 调度窗口保留**（L1 368/512、workspace 4 slot 都够）；**UB 按 2-head 驻留设计**（现状即是）。
只有当某项优化要求"UB 内同时放 4 份 head 工作集"（例如 head 级流水、V6 与 V3 真并行）时，
才把 `kHeadsPerGroup` 降到 **2**——代价是任务数 ×2、每任务固定开销（事件 alloc/release、prologue）摊薄变差；
收益是 L1 余量从 144 KiB 升到 288 KiB、UB 约束松开。

## 6. 与现状的差异清单（每项独立验证）

| # | 差异 | 目的 | 风险 | 验证 |
|---|---|---|---|---|
| **D1** | V6 由"C5 之后"提前到"V1 之后"（同回合） | V6 藏进 C2 窗口 | 共享区换义、GM 可见性 | L1 位级 + ≥3 轮 hang 看护 |
| **D2** | C4/C5/C7 合并到第二回合（ready/free 3→2 对） | 省一次跨核往返 | 集合同步配平 | 同上 |
| **D3** | 尾部 4-head 展开移植（A5 已有，A2/A3 未走） | 尾波均衡（培训同款手法实测 +1.10%/+4.55%） | GVA 组边界、尾块 | 位级 + 尾波比（每核 max/avg）<br>**内存可行性已由本预算确认**：尾部展开只改任务映射，不改每 head 工作集；仍需核对 `HV/HK` 整除 4、尾块 GVA 组不拆、AIC 与两个 AIV 尾部映射顺序一致 |
| **D4** | C7 的 RHS 预取（用 L1 余量 144 KiB） | 省 C7 等待 | L1 槽位冲突 | 位级 + L1 峰值 |
| **D5**（可选） | GVA 下 Q/K norm 按 HK 去重 | 省 `(1−HK/HV)` 的 norm 与重复读 | 跨 AIV 共享 | 位级 + `HV/HK` 消融 |

## 7. 待实测项与决策规则（决定 D1/D2 是否启用）

必须先补的 4 个数（分析计划 D2/D3，A2 真机、同机同频、3 warmup×10 取中位）：

| 量 | 怎么测 | 用途 |
|---|---|---|
| `V6` 时长 | 屏蔽差分（body 置空、同步不变） | 判断 V6 能否藏进 C2 窗口 |
| `C2` 时长 | 同上 | 同上 |
| `δ_flag` | 合并相邻回合的差值 / Insight 事件间隔 | 判断减一个回合值不值 |
| 尾波比（每核 max/avg 工作量） | 由 tiling 的任务分配直接算 + 实测时长交叉 | 判断 D3 的收益 |

**决策规则**：

- `min(V6, C2) ≥ 总时长 5%` ⇒ 采用 **D1**（V6 提前）；
- `δ_flag ≥ 总时长 3%` ⇒ 采用 **D2**（回合 3→2）；
- 任一条不满足 ⇒ **保持现状 3 回合调度**，本轮只做 **D3/D4**（两者与调度无关，收益独立）；
- D1/D2 若触发但 §4.2 的 4 条任一不成立 ⇒ 回退到现状并在 `design.md` 记录原因。

## 8. 验收判据

```text
功能：L1 dump BIT_IDENTICAL（逐位一致）；L2 smoke 10/10；L4 41/41；
      官方 ATK：200 条精度用例 + 432 条 TilingKey 用例（mss 内存检查当前暂停，不能声称 sanitizer 通过）
性能：A2/A3 目标待定；同机同频、msprof kernel duration、3 warmup×10 次取中位，
      基准 shape 用 ATK perf 的位置 7/8（B1 HK96 HV96 T8192 / T16384 varlen）
稳定：调度类改动（D1/D2）≥3 轮无 hang；flag set/wait 计数两侧严格配平
预算：UB ≤184 KiB；L1 ≤512 KiB；workspace 峰值与 host tiling 一致
```

## 9. 一句话结论

**Stage 划分维持 8 个不动**（V0/V1/C2/V3/C4/C5/V6/C7，边界由约束 1/2/12 唯一确定）；
**4-head 调度窗口保留**（L1 与 workspace 都够），**UB 侧按 2-head 驻留**（184 KiB 正好用满，4 份需 328 KiB 不可能）；
真正的性能空间在**调度与流水**：V6 提前（D1）、回合 3→2（D2）、尾部 4-head 展开（D3）、C7 RHS 预取（D4），
其中 D1/D2 需先用分 stage 计时与 `δ_flag` 实测按 §7 的规则确认后再改。
