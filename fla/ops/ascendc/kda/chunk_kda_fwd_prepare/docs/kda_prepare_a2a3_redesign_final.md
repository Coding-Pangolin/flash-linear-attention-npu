# ChunkKdaFwdPrepare（A2/A3）Stage 重设计：总览（基于 234 实测）

## 0. 收口状态（2026-10-09）

**设计结论（全部来自 192.168.9.234 实测）**

1. **Stage 边界 = 8 个**（`V0→V1→C2→V3→C4→C5→V6→C7`），**不动**；依据是数据依赖 +
   工作集 + 逐 Stage 可验证性（§4），**与 A5 对齐无关**。
2. **AIC 侧被 AIV 完全掩蔽**：把 `C2/C4/C5/C7` 全部空转，总时长 **+370 µs（不降反升）**；
   整核画像显示 **AIV 跨度 ≈ kernel 时长**（case 63/65/59 = 86%/95%/96%），AIC cube 只占 1~2.5%
   ⇒ **AIC 余量不是可用并行度**。
3. 因此重设计的着力点只在 **AIV 内部**，已落地并验证 4 项：
   `A1` 块状 norm（−36.7%）、`P3-A` norm 标量压紧（−416 µs）、`P3-B` cumsum log 扫描（−117 µs）、
   `P4` norm 逐行缩放改 Brcb+整块 Mul（−602 µs）
   ⇒ **18,848.7 → 10,797 µs（−42.7%）**，当前最优包 **`clean_lp4`**。
4. 其它候选**全部有实测结论**：A4′ 叶逆外迁 cube（+2,516 vs 收益 2,362 ⇒ 净亏）、
   P1 叶逆 Gather 两列合并（元素翻倍 +1,060 µs）、删同流水屏障（精度炸 3.4e11）、
   `E(G)` 契约去 exp（结构性错误：case 60/61 有 **37%/46%** 元素越过 ±126 的 fp32 指数边界）、
   V6 kg 族（556 µs 但结构性必需）。
5. **精度**：`clean_lp4` 在 **18 个 precision case** 上全部 PASS——含尾块 1/31/32/33 行、
   gate 四种模式、beta 三种模式、l2norm 小 eps、GVA（HK<HV）、B6 多批、以及模型规模的
   case 60/61/68/80。

**已知限制**：本容器的 ATK accuracy 通道是坏的（`mixed_tolerance_bm`），替代通道
`acc_ab.py` 对 17–20 号 varlen 用例造不出合法输入（op 报 161002 参数校验）
⇒ 变长精度覆盖目前只到"人工两段窗口"（case 61）这一档；补齐需要在能跑 ATK 的环境上做。

**未做的可选优化**（都已量化、都不属于边界/分工问题）：
`HardEvent::V_V` 块间软流水（V1Vf 藏 Exp latency，估 0.3~0.6 ms，需 3~5 次构建）；
把 V6 提前以省一次跨核回合（≈0.15 ms，但改动面大、收益小）。

> 单一入口文档。所有数字都来自 **192.168.9.234 / 8×910B3 / device 7 / 1800 MHz** 的实测，
> 详细证据在配套文档里（见 §8）。
> 用途：给下一个接手的人/agent 一份"已经确定什么、还剩什么、坑在哪里"的全局视图。

## 1. 一句话结论

**Stage 边界（8 个）经实测确认无需重切**；重设计的重点是**段内实现**：
AIV 上最贵的两段（逐行 L2 norm、32×32 叶子逆）都要在**原 stage 内**压下来。
其中 **块状 norm 已落地并验证（case 63 −36.7%）**；
**"叶子逆外迁 cube"（A4′）已按完整链路实测，结论是否定的**——cube 侧为完成
4 个 16×16 逆需要"每步 fixpipe 落 GM + Nd2Nz 回 L1"的串行往返（+2,025 µs），
再加 tile 生成与搬运（+491 µs），合计 **+2,516 µs > AIV 侧叶逆的 2,362 µs**，
净收益为负（详见 `kda_prepare_a2a3_A4_prime_probe_result.md`）。下一步应回到
**AIV 内部压叶逆**（Gather 实测占 1,264 µs）。

**同时纠正一条更重要的前提**：`design_a2a3_e2_stage_attribution.md` 里那张
"分 Stage 屏蔽差分"表（V0+V1 40.1% / V3 25.2% / V6 16.4% / C2 16.3% / C4+C5+C7 18.4%）
**各项之和 > 100%，不能当成本分解**。2026-10-09 用"**只屏蔽计算、保留全部核间回合**"
的直接对照测出：**把 AIC 的 C2/C4/C5/C7 全部空转，总时长反而 +370 µs（11,912 → 12,285）**，
整核画像也显示 **AIV 的跨度 10.17 ms ≈ kernel 总时长 11.88 ms**、AIC 的 cube 只有 264 µs
⇒ **AIC 侧完全被 AIV 掩蔽，余量不是可用并行度**。详见
`kda_prepare_a2a3_aic_hidden_evidence.md`。所以重设计的**唯一着力点是 AIV 内部的
向量指令条数与标量↔向量耦合**，不是 stage 边界、也不是 AIV↔AIC 的分工搬移。

## 2. 实测基线（case 63 = B1 HK96 HV96 T8192，warmup 2 / repeat 10）

### 2.1 整核画像（msprof op）

| 版本 | 任务时长 | AIV vec | AIV scalar | AIC busy（cube） |
|---|---:|---:|---:|---:|
| 原始基线 | 18,812 µs | 12,607 (67.0%) | 4,726 (25.1%) | 3,441 (21%)，cube 264 (1.4%) |
| A1 之后 | 12,148 µs | 7,872 (64.8%) | 4,294 (35.3%) | 3,466 (28.5%)，cube 264 |

⇒ A2/A3 上这个算子是 **AIV 受限**（vec 长期占 ~65%），**cube 长期空闲**；
`aiv_scalar_vector_stall` 从 6.2 ms → 0.86 ms（A1 的效果），`scalar_stall_by_ub` 几乎不变（5.8 ms）。

### 2.2 分 Stage 屏蔽差分（E2，调用点 `if(false)`）

> ⚠ **这张表只能当"上界提示"，不能当成本分解**：同一行的五项相加（14,203 µs）**大于**
> 基线总时长（12,184.7 µs）。2026-10-09 用"整侧空转"的直接对照测出
> **AIC 的 C2/C4/C5/C7 全部空转反而 +370 µs** ⇒ AIC 侧的活本来就被 AIV 掩蔽，
> 这张表里 C2 / C4+C5+C7 的 16.3% / 18.4% 不是可用收益。
> 见 `kda_prepare_a2a3_aic_hidden_evidence.md`。

| 版本 | `V0+V1` | `V3` | `V6` | `C2` | `C4+C5+C7` |
|---|---:|---:|---:|---:|---:|
| 原始基线 18,949.7 µs | 11,375 (60.0%) | 2,854 (15.1%) | 1,873 (9.9%) | 1,857 (9.8%) | 2,066 (10.9%) |
| **A1 之后 12,184.7 µs** | **4,892 (40.1%)** | 3,075 (25.2%) | 2,002 (16.4%) | 1,988 (16.3%) | 2,246 (18.4%) |

### 2.3 段内子屏蔽（把 max 项再切开）

| 项 | 测量方式 | 结果 |
|---|---|---|
| Q/K L2 norm（旧实现） | 属性 `use_qk_l2norm_in_kernel=False` | **8,620 µs（45.5%）** |
| Q/K L2 norm（新实现） | 同上 | **1,976 µs（16.2%）** |
| V1 指数链 | 屏蔽 V1 的 tile 循环 | 1,460 µs（12.0%） |
| chunk 内 cumsum（63 步串行） | 屏蔽该 for | 333 µs（2.7%） |
| gate 数学（V0+V6） | 块级屏蔽 | 299 µs（2.5%） |
| **32×32 叶子逆（2 个/chunk-head）** | 屏蔽两次 `SolveLeafRightLooking` | **2,362 µs（19.4%）** |
| 叶逆内部：Gather | 只关掉每步 Gather | **1,264 µs（10.6%）** |
| 叶逆内部：Mul/Sub+屏障 | 由上面两项相减 | ~1,098 µs（其中屏障 ~266 µs） |

## 3. Stage 划分的结论

1. **边界维持 8 个不动**（`V0 → V1 → C2 → V3 → C4 → C5 → V6 → C7`）：
   约束 1/2/12 唯一确定；且实测显示**去掉任一 stage 都真的省时间**（没有"白等"的边界）
   ⇒ 没有重切边界的理由。**"与 A5 同名同序"已按要求解除**，本条改为实测结论。
2. **执行单元归属要改**：AIV 侧最贵的串行段（叶子逆）应外迁到空闲的 cube；
   AIC 侧（C2/C4/C5/C7 + idle cube）有充足余量。
3. **4-head 调度窗口保留**（L1/workspace 够），**UB 侧仍是 2-head 驻留**（184 KiB 用满，4 份需 328 KiB）。

## 4. 已落地并验证的改动

### A1：`V0` 的逐行 L2 norm → 块状（候选包 `leafpair` / 基线 `normblk4`）

### P3：`V0` 的两处标量/串行开销已落地（2026-10-09，**−533 µs，−4.5%**）

### P4：norm 的逐行缩放改成 Brcb + 整块 Mul（2026-10-09，**−602 µs**）

当前最优包 **`clean_lp4` = 10,797.1 µs**（初始 18,848.7 → **−42.7%**）。
P4 把 norm 尾部剩下的 64 次标量 `GetValue` + 64 条逐行 `Muls`/`Cast` 换成
`Brcb` + `ReplicateRowScalarsToRepeat` + `MulTileByRowScalars`（本仓 V6 beta 路径
已用同一套 helper 验证）。数学不变；精度 60/61/80/13/26/68 全 PASS。
详见 `kda_prepare_a2a3_aiv_levers_evidence.md` §10（含"acc 读数有 1 ULP 抖动、
不能拿'逐位一致'当判据"的方法学更正，以及"源码快照必须紧跟刚构建的那份"这一教训）。

三处改动的拆分（case 63）：P3-A −416 µs、P3-B −117 µs、P4 −602 µs，
合计 **−1,135 µs（−9.5% vs `clean_lp`）**；`clean_lp3`（A+P4，缺 B）是中间态。

当前最优包 **`clean_lp2` = 11,398.9 µs**（初始 18,848.7 → **−39.5%**），两处改动都在 `V0Vf`：

1. **P3-A（−416 µs）**：去掉 norm 尾部"把 `WholeReduceSum` 结果压紧到另一缓冲"的
   64 次 `GetValue/SetValue`（外加一次 `V_S/S_V` 往返）——该结果本来就是**连续落位**，
   直接原地 `Adds(eps)→Sqrt→Div` 即可。
2. **P3-B（−109 µs）**：token 维 cumsum 从"63 步逐行 Add + 63 个 `PipeBarrier`"
   （= 63 次流水排空）换成 Hillis-Steele log 扫描（6 轮 × 2 条块指令 + 1 屏障），
   在 `g`（共享区）与 `tileBuf`（私有区 0x8000，本 Stage 已用完）之间乒乓。
   ⚠ `gk` 的求和顺序变成树形（末位 ~1e-7 差异）。

精度：case 60/61/80/13/26 **全 PASS，且 worst 值与改动前逐位相同**（4.883e-04 / 9.766e-04 /
4.883e-04 / 1.192e-07 / 2.384e-07）⇒ 树形舍入差异没有改变任何用例的最坏误差。
详见 `kda_prepare_a2a3_aiv_levers_evidence.md` §9。

做法：`Cast` 整块 → `Mul` 平方 → 折半 `Add` + `WholeReduceSum` 出行和 → 标量压紧一次 →
块状 `Adds/Sqrt/Div` → 标量读一次 → 逐行 `Muls/Cast`；`validRows < 32` 的尾块回退旧实现。

| 项 | 结果 |
|---|---|
| case 63 | 18,85x µs → **11,922.6 µs（−36.7%）** |
| 精度 | 11 条用例通过（含 GVA `HK<HV`、TND/varlen、尾块 1/32/60 行）；其中 9 条与基线**逐位同误差** |
| 附带微调 | 两个叶子逆合并到同一循环共享屏障（186→93 屏障，−266 µs） |

### 关键工程坑（都已验证并记档）

1. `WholeReduceSum` 的结果在 dst 上**连续**（不是 1 block/repeat）；按 block 读会串行错位；
2. **VEC 指令 UB dst 必须 32B 对齐**（对角写入要用"行首对齐基址 + mask 选块内 lane"）；
3. `Sub(dst,src0,src1)` 的 `src0` 不能是 `dst` 自己 + repStride=0；
4. UB→GM 的 `DataCopyPad` 的 **UB 侧 stride 单位是 32B block**（不是 byte）；
5. CANN `set_env.sh` 在 `set -u` 下会中断；`build.sh` 每次重建 third_party（~5 min/变体，勿并发）；
6. **只设 `ASCEND_CUSTOM_OPP_PATH` 不足以切换算子版本**：op store 还会跟 `opp/vendors/fla_npu_transformer`
   软链走 —— A/B 脚本必须同时改软链（否则两个版本量出同一个数）；
7. `kill -9` 掉持有设备的进程会让 tsd 变坏（`TsdOpen failed`），需等设备恢复再复用；
8. **诊断读数**：写 `args_.<output>` 能落到对应输出槽（`aqk`=索引 1、`w`=索引 3，已用特征常数验证）；
   但"同一 build 里连写多段"会被同一内容污染 ⇒ 一律**一次 build 只 dump 一段、只回答一个问题**。

## 5. A4′（叶子逆外迁 cube）：**2026-10-09 实测否定**（保留全过程作为证据）

> **结论（234 实测，perf case 63，W=2/R=10）**：
>
> | tag | 内容 | event (µs) | Δ |
> |---|---|---:|---:|
> | `leafpair` | 当前最优（无 tile、无 cube 求逆） | 11,922.8 | — |
> | `probe4c` | 4 个增广 tile + **cube 空转** | 12,413.5 | **+491**（AIV 侧代价） |
> | `probe4d` | 4 个增广 tile + cube **4 次** 16×16 逆 | 14,438.9 | **+2,516** |
>
> ⇒ cube 侧 4 次逆 = **+2,025 µs**，已经吃掉 AIV 侧叶逆的全部收益（2,362 µs），
> 而 32 级还要加跨块项与 X0/X1 装配，只会更亏。根因两条：① C220 的 fp32 **L0C 不能
> 直写 L1**，一次 16×16 逆要 4 个"fixpipe→GM→Nd2Nz→L1"串行往返；
> ② 本算子核间握手是**逐回合 lockstep**，AIC 的空闲**不能**吸收这些工作量。
> **所以"AIC 有 79% 余量"不构成外迁的理由**；8 个 stage 维持边界不变。
> 细节见 `kda_prepare_a2a3_A4_prime_probe_result.md`。

### 5.1 目标与收益

把 `V3` 里两次 32×32 单位下三角求逆（2,362 µs = 19.4%）从 AIV 外迁到 AIC；
**不新增跨核回合**（并进现有第 2 回合），AIV 侧顺带省掉跨叶修正与 Akk 打包。

### 5.2 算法（已 CPU 验证）

```text
16×16（或 8×8）严格下三角 N 的逆：X = [[-N, I],[0, I]]（增广），X^(2^m) 的右上块 = Σ(-N)^j = (I+N)^{-1}
  ⇒ 只需 log2(16)=4 次 MMAD（X²,X⁴,X⁸,X¹⁶）即可得 16×16 逆，**全程不需要"构造 I"**
32×32 由两个 16×16 对角块 + 跨块修正：X21 = -B⁻¹ (C A⁻¹)（与标杆同序）
```

CPU 验证（`validate_block_inverse.py`）：8×8 幂级数 4.5e-8、16×16 两级 1.7e-7（30 组随机），
bf16 rounding 后 28~30/30 组**逐位一致**。

### 5.3 前置（三块，均已落地并验证）

| # | 前置 | 状态 |
|---|---|---|
| ① | AIV 产出 L：payload 里 `L00`/`L11`（`kL00Shadow`/`kL11Shadow`）+ `L10`(=B) + relay 里的增广 tile | ✓ 精度 6/6 与基线同误差；case 63 时长不变 |
| ② | AIC 读 L：`L00`/`L11` `Nd2Nz` 进 L1 空闲区（`kAuxBase=0x60000`，Init 扩到 `kPeak+0x2000`） | ✓ case 60 误差与改动前逐位相同 |
| ③ | cube MMAD 场景：probe 做 3 连乘（装载/MMAD/fixpipe 全部照抄 `StageC4`） | ✓ 可编译/可跑；**代价仅 +0.25%（29.5 µs）**，几乎被流水掩盖 |

### 5.4 剩余步骤（下一步就做这个）

1. 在 `StageC4` 里把 16×16 两级方案接上（用已在 L1 的 `L00`/`L11`/`B`）：每个 16×16 块 4 次 MMAD，
   再做 32 级跨块 2 次 MMAD ⇒ 每个 32×32 逆 ~10 次 MMAD；
2. 让 `T = B@X0`、`q10 = -X1@T`、Akk 打包**消费 cube 的结果**（`V3a` 只保留 Aqk 解包 + L 生成 + 落盘）；
3. **判据用精度标杆**（不再依赖读回）：精度不变 ⇒ A4′ 成立；退化 ⇒ 按退化输出定位
   （W/U 错而 q_hat 对 ⇒ 逆不对；Akk 相关错 ⇒ 打包/落点不对）；
4. 通过后铺满"每叶 2×16 块 + 2 叶"，跑 ATK + perf（`m_a2.sh`）+ hang 看护。

### 5.5 备选（如果 cube 路线继续受阻）

在 AIV 内减串行：把"每步一次 Gather"换成"循环前一次性 Gather 成转置布局"（62 次 → 2 次），
上限约 1.0~1.2 ms（≈10%）；屏障已经做过一轮（只值 266 µs，不必再动）。

## 6. 其它已量化、未做的机会（**只列 AIV 侧**）

> 判据更新（2026-10-09）：**AIC 侧被完全掩蔽**（把 C2/C4/C5/C7 全部空转，总时长
> +370 µs），所以只有"减少 AIV 向量指令"的项才有意义；下表的上限来自屏蔽实测，
> 属于**上界**（屏蔽会连带拿掉内部 wait/set，读数偏高），真收益要用整侧对照或
> 端到端 A/B 复核。

| 项 | 上限 | 备注 |
|---|---:|---|
| **`V6` 自己的两个 exp 族**（输入是原始 gate、量级小、未饱和） | 2,002 µs（上界） | **现 P1**：唯一还没被证否的 exp 去重点；先逐族定价（`kda_prepare_a2a3_aiv_levers_evidence.md` §7） |
| ~~`V0` norm 标量压紧循环~~ | **416 µs** | ✅ **P3-A 已落地**（§"P3"） |
| ~~`V0` cumsum 63 步串行链~~ | **109 µs** | ✅ **P3-B 已落地**（log 扫描） |
| ~~`V0` norm 尾部的"逐行 Muls 标量"循环~~ | **602 µs** | ✅ **P4 已落地**（Brcb + 整块 Mul） |
| ~~`V1` 两个 exp 族去重（V0 契约改 `E(G)`）~~ | — | **❌ 结构性错误**：`G` 的 clamp ±126 就是 fp32 指数边界，case 60/61 有 **37%/46%** 的元素越过它，`E(G)` 单存会抹掉差值信息（同文档 §6） |
| ~~`V1`/`V6` 删同流水屏障~~ | −1,957 µs 但精度炸 3.4e11 | **❌ 屏障是本硬件上的语义**（同文档 §5） |
| `V3` 叶逆里的 `Gather`（62 次/chunk-head） | 1,264 µs（10.6%） | **2026-10-09 探针否决 P1 的"两列一批"**：Gather 按元素计费，合并会翻倍元素（`probe6`：+1,060 µs） |
| V3 叶逆的 mul/sub/屏障 | ~1.1 ms | 结构性下限：32× 广播是外积形式的固有代价（同文档 §3） |
| GVA 下 norm 按 HK 去重（`HK<HV`） | case 65 上限 ~31% | 只影响 GVA 形状 |
| 调度层（V6 提前 / 回合 3→2 / 尾部展开） | **≈0** | AIC 被掩蔽；不减少 AIV 向量工作量 |
| A4′（叶子逆外迁 cube） | **负**（+2,516 vs 收益 2,362） | 已实测否决 |
| cumsum（333 µs）、gate（299 µs） | — | 已量化：**不值得**再动 |

## 7. 复现与工具

```bash
# 编译+装到独立 vendor（不覆盖别人）
TAG=xxx bash build_a2.sh
# 计时（case 63 等）
JSON=<perf json> CACHE=<cache/case63.pt> TAG=xxx CID=63 W=2 R=10 bash m_a2.sh
# 精度（对 FP64 标杆，绕开 ATK 的坏通道）
TAG=xxx bash acc_batch.sh 60 61 62 68 80
# 整核画像
TAG=e1_xxx CID=63 bash run_msprof.sh
# 算法预验证（纯 CPU）
python validate_block_inverse.py
```

补丁（都可 apply/revert，含踩坑注释）：`patch_norm.py`（A1）、`patch_leafpair.py`、
`patch_lshadow.py`、`patch_aug.py`/`patch_aug2.py`、`patch_tileoff.py`、`patch_cubeprobe.py`、
`patch_cbcheck.py`、`patch_round2.py`、`patch_v3dump.py`/`patch_v3step.py`、`patch_mark.py`、
`patch_one.py`/`patch_one_fp32.py`、`patch_c4l.py`、`patch_tile2.py`、`patch_final.py`、
**`patch_probe4.py`/`patch_probe4x.py`（A4′ 全链路）、`patch_probe5.py`（整侧空转对照）**。

## 8. 配套文档

| 文档 | 内容 |
|---|---|
| `kda_prepare_a2a3_e2_stage_attribution.md` | E1/E2/子屏蔽/消融的**全部实测数据与判读** |
| **`kda_prepare_a2a3_aic_hidden_evidence.md`** | **AIC 完全被掩蔽的直接证据（含跨 3 个形状复核）+ E2 表不可加用的更正** |
| **`kda_prepare_a2a3_A4_prime_probe_result.md`** | **A4′ 铺满 4 个 16×16 块后的实测否决（含 UB 别名坑）** |
| **`kda_prepare_a2a3_aiv_levers_evidence.md`** | **AIV 侧杠杆：`clean_lp` 新基线、Gather"按元素计费"的探针（P1 否决）、剩余 exp 族定价计划** |
| `kda_prepare_a2a3_A4_leaf_inverse_to_cube.md` | A4′ 的**算法、tile 契约、cube 流水、前置落地、诊断过程与坑**（§4.x 全过程） |
| `kda_prepare_a2a3_stage_division.md` | 培训文档同格式的 Stage 划分最终稿（§5.3/§7.4 已按实测更新） |
| `kda_prepare_a2a3_stage_final.md` | 工程细节：落点表、调度与 flag 配平、空间预算、验收判据 |
| `kda_prepare_a2a3_budget_4head_vs_2head.md` | UB/L1/workspace 逐项字节账 |
| `kda_prepare_a2a3_opt_opportunities.md` | 8 个优化机会 + 优先级 + 培训"不要做"清单 |

## 9. 当前机器/仓库状态

- 234 上 `repo/` 的 `vec.h` / `cube.h` / `policy.h` **已还原到基线**（无 `probe4/probe5/E2B` 残留），
  OPP 软链指向 `e2v2_base`；源码快照在 `src_snap/inv2/`；
- vendor 包 63 个（**最优 `clean_lp4` = A1 + 两叶共享屏障 + P3-A/P3-B + P4，
  case 63 = 10,797.1 µs（−42.7% vs 原始 18.85 ms）**；
  `clean_lp3`（A+P4）/ `clean_lp2`（A+B）/ `clean_lp` / `probe9a` 是中间态；
  `probe4b/c/d`=A4′ 全链路、`probe5a/c`=整侧空转对照、`probe6`=Gather 计费、`probe8`=V6 kg 族）；
- 全部脚本/日志在 `/home/admin123/kda_a2/`；本文档与配套文档在仓库 `docs/` 下。
