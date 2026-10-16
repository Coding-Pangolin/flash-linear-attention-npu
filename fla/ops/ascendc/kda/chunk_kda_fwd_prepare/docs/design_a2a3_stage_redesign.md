# ChunkKdaFwdPrepare 在 A2/A3 上的 Stage 划分再设计（评审稿）

> 参照：《线性 Attention 算子设计培训：Stage 划分与 4-Head 调度》的 19 条 Stage 约束 + 设计流程 9 步。
> 现状基线：本仓 `fla/ops/ascendc/kda/chunk_kda_fwd_prepare`（`docs/design.md` §3 八阶段流水、
> `op_kernel/arch22/{cube,vec}.h`）。

## 0. 结论先行（3 条）

1. **八阶段划分本身是对的**（V0→V1→C2→V3→C4→C5→V6→C7），逐条对 19 条约束只有 1 处是"平台被迫"、
   2 处是"可以更好"。不建议把 stage 数继续合并（见 §2.1）。
2. **最值得改的是调度而不是划分**：`V6` 只依赖 `V0`（不依赖 C2/V3/C4/C5），把它从"C5 之后"
   提到"V0/V1 之后、V3 之前"，可以让 **V6 藏进 C2 的窗口**，并把跨核 ready/free 回合 **3 → 2**。
3. **A2/A3 必须接受一条与约束 4 的偏离**：C2 的 raw score 只能经 **GM relay** 交给 V3（无 L0C→UB）。
   这不是设计缺陷，要写进 design；真正该消除的重复搬运是 **GVA 下按 value head 重做 Q/K 归一化**（违反约束 16/10）。

## 1. 现状（A2/A3 实测口径）

| 项 | 现状 |
|---|---|
| 逻辑 Stage | V0(norm/beta/cumsum) → V1(Qplus/Kplus/Kminus) → C2(4 band MMAD) → V3(mask/beta/叶子逆) → C4(T=B@X0) → C5(q10=negX1@T) → V6(qg/kg/K_beta_g/V_beta) → C7(W/U) |
| 跨核回合 | **3 个 ready/free 回合**：① V0+V1 ⇄ C2 ② V3 ⇄ C4 ③ V6 ⇄ C7（C5 无跨核等待） |
| 4-head window | 每 AIC workgroup 最多 4 个 value head；**pair0={h0,h1}、pair1={h2,h3}**；AIV0 做 local h0/h2，AIV1 做 h1/h3（`localHead = pair*2 + aiv_`） |
| 同步 | mode 0x2 集合同步：AIC 一次 set 唤醒两个 AIV；两个 AIV 都 set 才唤醒 AIC。ready=0/1，free=2/3 |
| UB（A2/A3） | 192 KiB 硬件、**184 KiB 可用**（尾 8 KiB 保留）= 2×72 KiB 私有区 + 40 KiB 共享区（G 32 KiB + scratch 8 KiB） |
| L1 | 512 KiB，**4 个 head lane × 72 KiB**（C2 payload）+ X0/NegX1/T/Akk 各 16 KiB，峰值 368 KiB |
| workspace | 每 slot 105 KiB（33 KiB context + 72 KiB payload）+ arch22 追加 **20 KiB AIC 独占 relay**；每 workgroup **4 slot** |

## 2. 逐条对 19 条约束

### 2.1 已经合规（不用动）

| 约束 | 现状 |
|---|---|
| 1（每 stage 只能 cube 或 vector 之一） | V0/V1/V3/V6 vector；C2/C4/C5/C7 cube |
| 2（cube 不依赖本 stage 新产物） | C2 只读 V1 的 Qplus/Kplus；C4 读 V3；C7 读 V3+V6 |
| 3（L1 只给矩阵、UB 只给向量） | 严格遵守 |
| 4/5（中间量→UB 双份） | 2 私有区 + 2 compute slot（A5）；A2/A3 2×72 KiB 私有区 |
| 7/9（矩阵→cube 放 L1、**预留 4 份**） | L1 4 head lane；workspace 4 slot |
| 12（不分 pass/分 tile；一次搬完 + 一轮算完） | 每个 vector stage 各自一轮"同名向量函数"（A5 = RegBase VF；**A2/A3 无 VF**）；C2 一次提交 4 个 band |
| 15（少连续同型；有则任务要少） | **连续 vector V0→V1**：两者在**同一握手回合内**，跨核成本为 0；**连续 cube C4→C5**：C5 仅在 `M>32` 执行且无跨核等待 ⇒ 任务已足够少 |
| 17（L1/UB 不搬位） | policy 注释明确"payload 在原址换义"；C4 把 L1 head lane 从 score 换义为 B |
| 18（同语义不同 head 同操作） | GVA 组首 owner 写 q_hat/k_hat/rstd，避免 GM 重叠写 |

**为什么不把 V0/V1 合并成一个 stage**：在 A2/A3 上两者复用**同一段 72 KiB 私有区**
（`Q/K` 32 KiB + `gate/Kminus` 40 KiB），"合并"的内存代价与现在相同 ⇒ **不是容量问题**。
不合并的原因是 **A5 侧的一次 VF 约束**（一个 Stage 的数学要在一次 RegBase VF 内完成）加上 §4.4
要求两平台 Stage 函数同名同序（否则无法逐 Stage 对比 / A5-A2 对照）；约束 15 的担忧已被
"同一握手回合"消掉。⇒ **保持两个 stage**。

### 2.2 与约束 4 的被迫偏离（要写进 design，不是缺陷）

约束 4 要求"矩阵结果给后续 vector 用 ⇒ 放 UB"。但 **A2/A3 没有 L0C→UB 直连**（fixpipe 只能落 GM），
而 V3 是 vector（UB 语义）⇒ C2 的 4 段 raw score 只能经 **GM relay**（`kArch22CubeRelay`，20 KiB）。
当前实现正是如此，且 C4 复用同一 relay 前 4 KiB 放 T（原址换义，靠 ready/free 保证 WAR）。
**建议**：在 `design.md` 里显式标注这是"约束 14 允许的妥协 + 平台约束"，并把 relay 生命周期
（C2 写 → V3 读 → C4 覆写前 4 KiB）写成一张表；不要试图消除它。

### 2.3 可以更好的两处

**(A) V6 的位置：现在在关键路径上，应该藏进 C2**

`StageV6` 只读 `qhat/khat/betaEff`（workspace，V0 产物）、`gk`（GM，V0 产物）、`v`（输入），
**不读 C2/V3 的任何结果**。但当前调度把它放在 C5 之后，于是：

```
现在：V0+V1 ─ C2 ─ V3 ─ C4 ─ C5 ─ V6 ─ C7        （6 段串行，3 个跨核回合）
建议：V0+V1 ─┬─ V6 ─────────┐
             └─ C2 ─ V3 ─ C4 ─ C5 ─ C7            （V6 与 C2 重叠，跨核回合 3 → 2）
```

- 关键路径从 `V0V1 + C2 + V3 + C4C5 + V6 + C7` 变为 `V0V1 + max(V6, C2) + V3 + C4C5 + C7`；
- 因为 V6 提前完成，C7 不再需要单独的 "V6 ready" 回合 ⇒ **第二回合可一次做完 C4→C5→C7**；
- 需核对的 4 件事：① V6 的 GM 输出（K_beta_g/V_beta）要对 AIC 可见（`DataSyncBarrier`/FIX 事件）；
  ② UB 私有区 V6 与 V3 顺序执行、不并行（同 AIV）；③ 共享区（G/scratch）V6 先于 V3 使用；
  ④ flag 计数从 3 对降到 2 对，两侧 set/wait 必须严格配对（否则 hang）。

**(B) GVA：Q/K 归一化按 HK 去重（约束 16/10）**

design §5 已承认："当前实现仍按 value-head slot 搬运和归一化 Q/K，多个 value head 映射到同一
Q/K head 时会从 GM 重读对应 Q/K"。这是 **按 HV 重复做 norm 运算**（约束 16 要求重复运算放驻留区）。

- 建议：norm 只在 **GVA 组首 owner** 做一次，结果（q_hat/k_hat/q_rstd/k_rstd）按 HK 写 workspace；
  其余 value head **直接读归一化结果**（GM→UB），跳过平方和/rsqrt/缩放；
- 这仍是"同语义同操作"（所有 head 都从 GM 取归一化结果），不违反约束 18；
- 收益与 `HV/HK` 成正比（`HV/HK=4` 时省约 3/4 的 norm 向量工作量与 Q/K 的重复计算）；
- 搬运量不增（q_hat/k_hat 本来每个 head 也要读，因为 `qg = q_hat*E(G)` 是 per-value-head 的）。

## 3. 推荐的 A2/A3 Stage 划分与调度

**Stage 划分（与现在相同，8 个逻辑 stage）**，只改调度归属：

| 回合 | 执行单元 | Stage | 数据路径 |
|---|---|---|---|
| R1 | AIV（每 pair 一回合） | V0 → V1 → **V6** | V0/V1 产物：Qplus/Kplus/Kminus 落 **L1 head lane**（C2 用）+ workspace payload；V6 产物 K_beta_g/V_beta/qg… 落 **GM**（C7 与输出用） |
| R1 | AIC | C2 | L1 head lane 读 Qplus/Kplus → 4 band MMAD → raw score 写 **GM relay** |
| R2 | AIV | V3 | 读 GM relay 的 raw score + beta_eff → Aqk/B/X0/X1/negX1/Akk（Akk 写 **workspace payload**） |
| R2 | AIC | C4 → C5 → C7 | C4: T=B@X0 落 **L1**；C5: q10=negX1@T 落 **L1**（仅 `M>32`）；C7: W/U = Akk@[K_beta_g\|V_beta]，读 **GM** 的 Akk / K_beta_g / V_beta |

**约束对照**：C2/C4/C5/C7 之间是 cube→cube（约束 7：中间量放 L1；C4 的 T 与 C5 的 q10 都在 L1，
不需要跨核 flag）；V3→C4/C5/C7 是 vector→cube（约束 6：经 GM，Akk 与 K_beta_g/V_beta 都已在 GM）。

**预算复核**：UB 与现在同（V6 与 V3 原址复用私有区，顺序执行）；L1 峰值不变（4 head lane + X0/NegX1/T/Akk
= 368 KiB ≤ 512 KiB）；workspace 不变（4 slot × 0x1F400）。

## 4. 落地顺序（每步可独立验证）

| 步 | 动作 | 验证 |
|---|---|---|
| 1 | 在 `design.md` 补：① C2→V3 的 GM relay 偏离说明与生命周期表 ② V0/V1 为何不合并 ③ 本表 | 文档评审 |
| 2 | **V6 提前**（调度改动，不动数学）：AIV 侧 `V0/V1/V6` 同回合，AIC 侧 C4/C5/C7 合并到第二回合 | L1 位级同前 + 全门禁；`.o` 允许变但数值必须逐位一致 |
| 3 | GVA norm 去重（HK 级缓存） | GVA 用例位级 + `HV/HK=2/4` 消融，确认 norm 工作量下降、总时长不升 |
| 4 | 4-head window 消融（1/2/4/8 head） | 按培训 §7.4 的口径：同机同频、3 次 warmup + 10 次采样中位；记录尾波与 L1/UB 占用 |

## 5. 必须一起测的边界（培训 §7.6）

`H` 不是 4 的倍数（尾窗口 `headEnd=min(begin+4,H)`）、`M<=32`（跳过 C4/C5）、`M=64` 整块、
变长（`cu_seqlens` 非均匀）、GVA（`HK<HV` 且不整除 4 时禁用尾部 4-head 展开）。
