# 重构迭代规程：怎么改才"零功能、零性能"漂移

> 2026-09-29。配套 [`refactor_plan.md`](refactor_plan.md)（改什么）与
> [`switch_remediation_plan.md`](switch_remediation_plan.md)（开关明细）。
> 本文只回答一件事：**每一步怎么验、什么时候可以只验一半、什么时候必须全量重测。**

## 0. 一条原则：能证"机器码相同"的批次，绝不跑性能

重构的风险只有两种：**功能变了**、**性能变了**。判据有三层，从强到弱、从便宜到贵：

| 层 | 判据 | 代价 | 结论强度 |
|---|---|---|---|
| **L-A 机器码恒等** | 同一 `SOC`、同一宏取值下，kernel `.o` **逐字节相同**（`md5sum`） | 1 次编译（~1 min） | **最强**：同二进制 ⇒ 功能与性能**必然**一致，不需要跑门禁/性能 |
| **L-B 数值恒等** | `dump_hm` 的 L1 位级比对 `BIT_IDENTICAL`（逐元素完全一致） | 门禁 ~1 min + 采性能 | 功能等价（含 bitwise），但**性能可能变**（调度/指令数变了） |
| **L-C 语义等价** | L2 smoke（`max_abs≤0.05`）+ L4 41 条 + L3 探针 | 门禁 ~5 min | 允许舍入级差异；仅在**必须改数值路径**时使用（本重构计划里**不应该出现**） |

**推论（本次重构的全部省力空间都来自这里）**：

- 纯删除/改名/格式化/搬运（T1、T2、T3、T3b 的一部分）→ 走 **L-A**，`.o` 相同就直接过；
- 引入 traits / 改 TilingKey 模板 / 事件 id 命名化（T3c、T4、T5）→ 必须 **L-B + 性能 A/B**；
- 只要出现 **L-C**（数值不再逐位一致）就说明**改动越界了**（重构不该改数值），按 §5 回退定位。

## 1. 每批次要过哪几层（照表执行）

| 步 | 期望 `.o` | 必须 L-B 位级 | 必须性能 A/B | 必须 soak | 备注 |
|---|---|---|---|---|---|
| **T1 清场**（删死文件、更正过时文档） | **相同** | — | — | — | 只删 `*.bak/*.txt/*.wip` 与非源码文档 |
| **T2 注释与格式**（删 118 行开发注释、clang-format） | **相同** | — | — | — | 若 `.o` 变了 ⇒ 说明动到了代码而非注释 |
| **T3 拆文件**（机械搬运，零逻辑） | **相同** | — | — | — | 唯一允许的差异是 include 顺序；用 `arch_view_diff.py` 佐证 |
| **T3b 文件头三张表** | **相同** | — | — | — | 纯注释 |
| **T3c 计算层重构**（traits/CubeChunkStateGemm/函数移出类/拆 Stage） | 允许不同 | **必须** | **必须** | **必须**（碰事件与槽位） | 最重的一步；建议按"角色 × 阶段"再分小批 |
| **T4 派发与开关**（`_tiling_key.h` 替 `TILING_KEY_IS`、`TORCH_MODE` 改名、删 0 分支） | 允许不同 | **必须** | **必须** | **必须** | 阻塞项；多个子批各自独立 commit |
| **T5 常量/换算/事件生命周期集中** | 允许不同 | **必须** | **必须** | **必须** | 事件 alloc/release 一变就要 soak |
| **T6 文档同步 / T7 用例与精度** | 相同（无代码改动） | — | — | — | 只改 docs/tests |
| **T8 交付清单** | 相同 | — | — | — | 收口 |

> 判据的"必须"是**下限**：如果 L-A 意外成立（`.o` 相同），可以跳过后两层并直接过——这是最理想的情况。

## 2. 单批 SOP（固定 9 步，可直接照抄）

```bash
OP=<算子目录>; TAG=t3c_a; BASE=t3a_base        # BASE = 上一批通过时的 TAG
```

1. **写清设计**：改什么 / 依据（`refactor_plan.md` 的哪一条）/ 期望 `.o` 是否变 / 回退方式。
2. **留基线**（每批开始前，若 BASE 不存在）：
   `TAG=$BASE bash scripts/gates/exp_switch.sh <soc> $REPO` —— 它会编译、装到 `opp_vendors/$BASE`，
   并打印 **kernel `.o` 的 md5**（存进本批记录）。
3. **单变量改动**：一批只做一件事；拆分结构时"只搬不改"。
4. **编译+切换**：`TAG=$TAG bash scripts/gates/exp_switch.sh <soc> $REPO`
   → 记录本次 `.o` md5，与 BASE 比：**相同 = L-A 通过**。
5. **门禁**：`TAG=$TAG BASE=$BASE bash scripts/gates/run_gate_all.sh "$OP"`
   → 要求 L0 PASS / L1 `BIT_IDENTICAL` / L2 10 项 / L4 41 条 / L3 探针干净。
6. **性能 A/B**（仅在 `.o` 不同时）：`python3 scripts/gates/perf_ab.py --tag $TAG --compare`
   → 判据见 §3；**必须同卡同频、同口径**。
7. **soak**（仅碰同步/落点/事件时）：`PROCS=20 PAR=4 DEV=<空闲卡> TAG=$TAG bash scripts/gates/race_soak.sh "$OP"`
   → 要求 `SOAK_CLEAN`。
8. **审计**：`bash scripts/gates/switch_audit.sh "$OP"` + `bash scripts/gates/convention_audit.sh "$OP"`
   → 记录 FAIL/WARN 数（只应随着批次单调下降）。
9. **提交**：commit message 用 §4 模板；把 4 组数字（`.o` md5、L1 结论、性能中位、soak 结论）写进 commit body。

## 3. 性能判据（怎么算"没退化"）

### 3.1 口径（必须固定，否则数字不可比）

| 项 | 规定 |
|---|---|
| 机器/卡 | **同一台机器的同一张卡**；记录 `npu-smi` 的型号、频率、温度 |
| 采样 | 每个 shape **重复 3 次取中位**（沿用计划 §5.2）；每次 `msprof op --launch-count=1 --warm-up=1` |
| shape 集 | **至少三个**：`T=1024/HV=8`（小）、`T=4096/HV=8`（中）、`T=11264/HV=32`（模型 case）；**KDA 变体单独一组**（`VARIANT=kda`），因为它与 GDN 的每 chunk 成本差 ~2× |
| 口径换算 | `msprof` 单次发射 ≈ **稳态 device event 的 1.27×**（已实测）；**基线必须用同一种采法**，不要跨口径比较 |
| 负载 | 采集期间**不跑其他任务**；同一批次内 BASE 与 TAG 交替采（A/B/A/B）以抵消频率漂移 |

### 3.2 判据（`perf_ab.py` 会自动判）

```
回退阈值：  median(TAG) > median(BASE) × (1 + PERF_TOL)      且  差值 > 3 次采样的极差
PERF_TOL 默认 2%（可调）；两项同时满足才算 REGRESS，避免把噪声当回退。
另外必须看单项 busy 分布（msprof --aic-metrics）：若总时长没变但 AIV scalar/vec 明显上升，视为隐性退化。
```

**性能优于基线不设上限**（重构中出现小幅变快是允许的），但要在 commit body 里解释原因（通常是删掉了一个屏障/一次搬运）。

## 4. commit 模板（每批一条，便于回溯）

```
refactor(ppfm <批号>): <一句话改了什么>

- 依据：docs/refactor_plan.md <T?> / <规范条款>
- .o  : <SOC> <EXTRA_MACROS> md5 <base8> -> <tag8>  [IDENTICAL | DIFFERS]
- L1  : BIT_IDENTICAL | <不适用（.o 相同）>
- 门禁: L0 PASS / L2 10/10 / L4 41/41 / L3 0/30
- 性能: T=1024 212.2->212.0us  T=4096 623.6->620.1us  model 3828.7->3820.0us (median of 3)
- soak: SOAK_CLEAN 0/20 | <不适用>
- 审计: switch FAIL a->b / convention FAIL c->d
- 回退: git revert <sha>
```

## 5. 失败处理（三种情形，别混）

| 现象 | 含义 | 处理 |
|---|---|---|
| 期望 `.o` 相同但**不同** | 改动越界（不是纯注释/搬运），或宏展开变了 | 先 `arch_view_diff.py` 对比展开后的源；确认是"允许变"的批次就改判为需 L-B+性能；否则拆小重做 |
| `.o` 不同且 **L1 位级不等** | **数值行为变了** —— 重构不允许 | **直接回退**这一个小批，二分（按函数/按 Stage）定位到具体改动 |
| `.o` 不同、L1 位级一致，但**性能回退** | 代码生成或调度变了（典型：屏障位置、事件顺序、buffer 别名） | 先看 `--aic-metrics` 的 busy 分布定位（vec/scalar/mte2/fixpipe 哪一项涨）；能归因就修，不能归因就回退——**重构不背性能优化**，性能问题另开一轮 |
| soak 出现 `SOAK_DIRTY` | 同步协议被动过 | 按 `race_status_20260929.md` 的触发条件排查；**保留**该批的开关做 A/B，改不动就回退 |

## 6. 环境与预算

1. **同机同卡**：每个批次固定一台机器一张卡（950 用 246/247，A2 用 221；A3 只能构建级）。
   跨机只允许"确认趋势"，不允许当作 A/B 判据。
2. **时间预算（实测口径）**：`.o` 对比 ~1 min；全门禁 ~5 min；性能一组（3 shape × 3 次）~15–20 min；
   soak（20 进程 PAR=4）~20 min。⇒ **只有 T3c/T4/T5 需要"门禁+性能+soak"全套（约 45 min/批）**；
   T1/T2/T3/T3b 走 `.o` 恒等，**每批 ≤5 min**。
3. **切回随时可用**：`exp_switch.sh` 把每批的 OPP 包留在 `opp_vendors/<TAG>`，
   回退只需把软链指回 BASE（或 `TAG=` 还原），不必重新编译。
4. **不要并行做两批**：一次只改一个变量，否则 `.o`/性能差异无法归因。

## 7. 工具清单（都在 `scripts/gates/`）

| 工具 | 用途 | 关键判据 |
|---|---|---|
| `exp_switch.sh` | 编单算子 OPP 包 + 切 TAG + 打印 `.o` md5 | `.o` 恒等（L-A） |
| `dump_hm.py` / `cmp_hm.py` / `run_gate_all.sh` | L0/L1/L2/L4/L3 门禁 | `BIT_IDENTICAL` + 41/41 |
| `perf_ab.py`（本文新增） | 性能基线保存与 A/B（3 次中位 + 阈值 + 极差保护） | `PASS/REGRESS` |
| `race_soak.sh` / `race_probe_cached.py` | 进程级竞态 soak | `SOAK_CLEAN` |
| `switch_audit.sh` / `convention_audit.sh` | 实验化痕迹与工程形态审计 | 数字单调下降 → 最终 `*_CLEAN` |
| `arch_view_diff.py` | 宏展开后的平台视角 diff（拆文件/改宏时证明等价） | 展开后逐行等价 |
