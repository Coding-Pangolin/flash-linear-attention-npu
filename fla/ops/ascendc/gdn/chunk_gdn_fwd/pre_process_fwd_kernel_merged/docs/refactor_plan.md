# 分步重构计划：对齐本仓《算子工程结构规范》与 PR#728

> 2026-09-29，基于 `2cc812a`（R20）。
> 目标：**保留已实测最优路径、不改变任何已验收的数值行为**（每步都过 L1 位级 + L2/L4；
> 碰同步/落点的步骤加跑进程级 soak），把代码整理成**本仓规范认得的形态**。

## 0. 依据（以本仓规范为准，skill 规范为辅）

| 优先级 | 来源 | 关键条款 |
|---|---|---|
| ① | 本仓 `docs/agents/reference/04-operator-development/engineering-structure.md`（**V1**；PR#728 = `refs/pull/728/head = e089ceff` 是最新版） | §2 标准目录结构；§3 host；**§4.1 根目录与 arch22/arch35（含 §4.1-8 编译路径开关命名）**；**§4.2 TilingKey 模板化是必需件**；§4.3 入口只接线；**§4.4 kernel 四层可读性结构 + 事件生命周期**；§7 编码细节；§8 交付前校验清单 |
| ① | 样板算子 `chunk_gated_delta_rule_bwd_finalize/op_kernel/`（规范点名的"kernel 可读性结构样板"） | `*.cpp` 77 行薄入口 + `arch35/{_common.h(106), _struct.h(87), _cube.h(1736), _vector.h(2422)}` |
| ① | PR#728 改的示例 `docs/agents/.../engineering-example/L2独立算子示例/` | cube 侧模板：**`OpNameCubePrimitives<DT>` traits + `CubeChunkStateGemm` + 文件头三张表 + 事件 id 与物理槽一一对应**；目录 `op_kernel/{op_name.cpp, op_name_common.h, arch22\|arch35/{_struct,_cube,_vec}.h}` |
| ② | `cannbot-skills` 的 `ops/ascendc-code-review/references/{cpp-style.md, ascendc-op-conventions.md}` | CANN C++ 风格（规则 3.3 禁开发阶段注释）；Kernel 禁 `TILING_KEY_IS`、禁 `_apt`、REG_OP 要 `InferDataType`（与 §4.2 同一问题，互为佐证） |

> 注：`cannbot-skills` 仓库里的 MR728（"add linear attention experience guidance"）与本 PR **无关**，仅其 LA 门禁可作补充参考。

**两条硬冲突（阻塞级，必须先改）**：

1. §4.2「**每个 Ascend C 算子都必须有 `op_kernel/<算子>_tiling_key.h`**，用 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL`；**不允许用"单实例 + 运行期分支"代替模板化**」—— 我们**没有** `_tiling_key.h`，且 kernel 里用 `TILING_KEY_IS(1/2/3)` 运行期分派（3 处）。
2. §4.4「**函数不写进类/结构体**；`<算子>CubeContext/VectorContext` 只放数据」—— 我们现在是 `class PpFwdVector` / `class PpFwdCube`，**所有函数都是类成员**。

## 1. 目标目录结构（对齐 §2 + PR#728 示例）

```text
fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/
|-- CMakeLists.txt                     # 已有
|-- README.md                          # 能力/输入限制/输出布局（唯一定义来源）
|-- docs/{api.md, design.md}           # 已有（design 要补 §4.4 的 Stage/布局/同步三张表）
|-- op_host/
|   |-- CMakeLists.txt
|   |-- pre_process_fwd_kernel_merged_def.cpp
|   |-- pre_process_fwd_kernel_merged_tiling.cpp/.h        # 瘦身：只留校验与入口
|   |-- pre_process_fwd_kernel_merged_tiling_processor.h   # ★新增：tiling 算法主体（header-only）
|   `-- op_api/...
`-- op_kernel/
    |-- pre_process_fwd_kernel_merged.cpp                  # ★瘦身成薄入口（~80 行）
    |-- pre_process_fwd_kernel_merged_struct.h             # 根目录兼容头：include archXX 结构
    |-- pre_process_fwd_kernel_merged_common.h             # ★新增：平台无关常量、同步协议声明、ChunkInfo/offset 换算
    |-- pre_process_fwd_kernel_merged_tiling_key.h         # ★新增（必需件）：ASCENDC_TPL_ARGS_DECL/SEL
    |-- arch22/{_struct.h, _cube.h, _vec.h}                # A2/A3
    `-- arch35/{_struct.h, _cube.h, _vec.h}                # A5
```

`archXX/_cube.h`、`archXX/_vec.h` 内部按 **§4.4 的五段顺序**摆（读者靠顺序定位）：

```text
① 文件头注释块      Stage 表 + L1/L0（或 UB）布局表 + 同步协议表（flag 名/方向/背压来源）
② 类型与 layout 层  <算子>CubePrimitives<DT>：TileCopy*/TileMmad/ElementAccumulator/
                    CopyL1ToL0*/CopyL0CToDst + FixpipeConfig + tla::MakeLayout 常量 + static_assert(L1/L0 上限)
③ 计算层            AIC: CubeChunkStateGemm(...) 一次调用完成「K 分块 L1→L0 + TileMmad 累加
                    (unit flag 表达首/末块) + L0C 搬出」；
                    AIV: StageNVf(...)（arch35 必须 __simd_vf__ + MicroAPI VF 融合；arch22 同名函数普通向量指令）
                    两者都只碰本地 buffer、不搬 GM、不发同步事件、尾块只传 validLen
④ 数据结构体        <算子>CubeContext / <算子>VectorContext —— 只放数据（张量/事件 id/只读状态/游标/模板别名）
⑤ 行为层            Init<角色> → Process<角色>（事件预置 + 任务主循环 + 按序调 Stage）→ StageN… →
                    CloseAndReleaseEvents；全部文件作用域 inline，第一参数是 ④ 的数据；
                    阶段函数头固定写"输入 / 输出 / 复用 / 同步"四行
```

## 2. 审视结果：我们 vs 规范（`scripts/gates/convention_audit.sh` 实测）

| 规范条款 | 要求 | 我们的现状（证据） | 处置 |
|---|---|---|---|
| §2 目录结构 | `_tiling_processor.h`、`_common.h`、`_tiling_key.h`、`archXX/{_struct,_cube,_vec}.h` | **单文件 `op_kernel/pre_process_fwd_kernel_merged.cpp` 2335 行**；无 `_common.h`/`_tiling_key.h`/`arch22`/`arch35`；3 个死文件（`*.cpp.bak`/`*.txt`/`*.wip`） | **T1/T3** |
| §4.1 布局 | 只允许"默认+A5 差异"或"A2/A3 与 A5 各一份"；arch 选择宏写在入口 `.cpp` 顶部 | 只有一份实现，内联 `#if PPFM_ARCH_IS_950` 散在正文（15 处条件点） | **T3** |
| §4.1-8 开关命名 | 编译路径开关**按场景命名**，禁泛化负向宏（点名 `TORCH_MODE`）；同一算子不允许两套开关并存 | 入口有 `#ifndef TORCH_MODE`；正文另有 19 个 `PPFM_*` 开关 | **T4** |
| §4.2 TilingKey | `_tiling_key.h` **必需**；`ASCENDC_TPL_*` 声明+枚举；内部 `if constexpr`；禁运行期分支判断编译期常量 | **无 `_tiling_key.h`；`TILING_KEY_IS(1/2/3)` 运行期分派**（第 2303/2313/2323 行） | **T4（阻塞）** |
| §4.3 入口 | 一个入口；只做地址解析与分派 | 入口 2260+ 行，含全部 Stage 计算与同步 | **T3** |
| §4.4 ① 文件头三张表 | Stage 表 / 布局表 / 同步协议表 | **没有**（只有散落的 `ITER*/R*` 注释） | **T3b** |
| §4.4 ② 计算层 | AIC：`CubePrimitives` traits + `CubeChunkStateGemm` + `static_assert` 空间上限；AIV：`StageNVf`（arch35 必须 VF 融合） | 手写 `RunTiledNT/TA`（L1→L0→Mmad→搬出）**未用 traits**、**无 `static_assert`**；AIV 用普通向量指令（无 `__simd_vf__`） | **T3c** |
| §4.4 ③ 结构体 | 只放数据，不放函数 | `PpFwdVector`/`PpFwdCube` 是**类，函数全在类里** | **T3/T3c** |
| §4.4 ④ 行为层 + 事件生命周期 | 结构体外 inline；`Init` 只接线/分 buffer/派生只读状态；`Process` = 事件预置 + 任务循环 + 调 Stage + 收尾；**Init 里按 slot `AllocEventID`+`SetFlag` 开首轮，Process 末 `WaitFlag` 闭环 + `ReleaseEventID`**；事件 id 与物理槽一一对应 | 事件用 `EVENT_ID0..5` 零散硬编码、无条件表；alloc/release 未集中 | **T3c/T5** |
| §4.4 ⑤ 命名与常量 | 阶段函数带 Stage 号；workspace **按语义命名**（禁 `ws0`/`offset+32768`）；尺寸常量带数值后缀；实现里不写裸数字；任务换算封 `GetChunkInfo(...)`、偏移封 `GetWorkspaceChunkOffset(...)` | workspace 用 `WS_M_F32/WS_H_BF/...` 裸偏移；`chunk` 换算内联；`0x4/0x2`、`16`、`32768` 等裸值散落 | **T5** |
| §4.4 同步协议集中 | 方向/flag 数量/复用与背压规则写在 `common.h` 顶部一次说明 | `kFlagInputs/Half1/VNew/DH` 与 `PPFM_SUBFLAG_STRIDE` 定义在 `.cpp` 内，无背压说明 | **T5** |
| §4.1 平台差异不改骨架 | 两份实现类名/接口/Stage 函数同名 | 现在只有一份（内联分档） | **T3** |
| §7 编码细节 | 行宽 ≤120；固定后缀（`_struct.h`/`_common.h`/`_tiling_key.h`）；include guard；报错带实参；常量集中 + `static_assert`；不提交生成物；按 `.clang-format`；三份文档与代码同步 | 行宽 0 超限 ✅、文件头 ✅、无 TODO ✅；但**118 行开发阶段注释**（`ITER*/R*`/待上机/实测负收益）、22 处装饰符、无 `static_assert` | **T2/T5** |
| §8 交付前清单 | 17 条逐项确认 | 需逐条补 | **T8** |
| 补充（cannbot-skills） | Kernel 禁 `TILING_KEY_IS`、禁 `DeviceGemm`/自实现矩阵乘、禁 include 自身 tiling 实现 | 与 §4.2 同一问题；无 `DeviceGemm` ✅；只 include `_struct.h` ✅ | 随 T4/T8 |

## 3. 分步计划（每步一个 commit，可独立回退）

| 步 | 动作 | 产出 | 验证 | 回退 |
|---|---|---|---|---|
| **T0 冻结** | 建分支；跑 `TAG=refactor_base` 全门禁 + 两个审计脚本存档；**先出 A2 的 UB bank 结论** | 基线 dump、`AUDIT_before.txt` | 全绿 | — |
| **T1 清场（纯删除/纯新增）** | 删 3 个死文件与 2 个 `*.txt` 原型；更正已过时的 `docs/a2_opt_status.md` | 干净的 `op_kernel/` | 无代码改动 | revert |
| **T2 注释与格式** | 删 118 行开发阶段注释（结论搬进 `docs/perf/`、`validation.md`）；去装饰符；对改动文件跑 `clang-format` | 只有注释/空行变化 | **`.o` 逐字节相同** + 全门禁 | revert |
| **T3 拆文件（机械搬运，零逻辑改动）** | 按 §1 拆：入口瘦身 → `_common.h` → `arch22\|arch35/{_struct,_cube,_vec}.h`（先 A5，再补 A2） | §1 目录树 | 同 arch/宏下 **`.o` 逐字节相同**（`arch_view_diff.py` 辅助） | revert |
| **T3b 补文件头三张表** | 每个 `archXX/{_cube,_vec}.h` 顶部写 ① Stage 表 ② 布局表（偏移/大小/内容/生命周期） ③ 同步协议表（flag 名/方向/背压来源）；`_struct.h` 偏移常量与表逐行对应 | 三张表 | 人工逐行对照 + 全门禁 | revert |
| **T3c 计算层重构** | AIC：`<算子>CubePrimitives<DT>` traits + `static_assert` L1/L0 上限，`RunTiled*` 收敛为 `CubeChunkStateGemm(...)`；AIV：函数移出类、拆 `StageNVf` 与编排层 `StageN…`；结构体只留数据 | `_cube.h`/`_vec.h` | **L1 位级** + 全门禁 + **soak 0/20** | 每角色单独 commit |
| **T4 派发与开关（阻塞项）** | ① 新增 `_tiling_key.h`（`ASCENDC_TPL_ARGS_DECL/SEL` 表达 gateMode×dtype 档），**替掉 `TILING_KEY_IS`**，host `#include` 同一份枚举；② `TORCH_MODE` → `FLA_TORCH_EXTENSION_INLINE_BUILD` 并写清语义；③ 按 `switch_remediation_plan.md` 删已定稿 0 分支、arch 差异收进 `archXX/_struct.h` 常量 | `_tiling_key.h`、常量表 | 编译（950+A2/A3）+ L1 位级 + L2/L4 + **soak 0/20** | 子项分开 commit |
| **T5 常量/换算/事件表集中** | 尺寸常量带数值后缀；workspace 按语义命名 + 注释生命周期（`// S0 kbg -> S12 doG`）；`GetChunkInfo(...)`/`GetWorkspaceChunkOffset(...)` 封函数；事件 id 与物理槽一一对应，Init 预置首轮、Process 末闭环 + `ReleaseEventID` | `_common.h`、`archXX/_struct.h` | L1 位级 + 全门禁 + soak（碰事件） | 子项分开 commit |
| **T6 文档同步** | `design.md` 补三张表与规则版本；`api.md` 补可选输出/返回码；`README.md` 补能力与输入限制；性能证据迁进 `docs/perf/round_NNN/`（PRE/POST、单变量、瓶颈归因、`baseline_status`） | 四份文档 | §8 清单逐条打勾 | — |
| **T7 用例与精度** | 覆盖矩阵按 §3.6/§6.1（与 Δ5 一致）重建：≥8 例、必需维度齐全；本算子契约锁死 `K=V=128`/`chunk=64` ⇒ `V=256/BT=128` 写明"契约外"；`tests/atk/<算子>/` 三份 JSON + yaml + gen + executor 齐备；TilingKey 覆盖表给实际选择证据 | `tests/atk/...` | 用例齐 + 精度全绿 | — |
| **T8 交付清单收口** | 逐条过 §8 十七条；两个审计脚本 CLEAN；`git status --short` / `git diff --check` 干净 | 审计输出 + 清单 | 两平台全门禁 + soak | — |

依赖：**T0 → T1 → T2 → T3 → T3b → T3c → T4 → T5 → T6 → T7 → T8**。
**T2（降噪）与 T3（搬运）都不改逻辑**，是后续 diff 可读的前提；**T4 是唯一的阻塞级规范冲突**。

## 4. 验证命令盒

```bash
OP=<算子目录>
bash $OP/scripts/gates/switch_audit.sh    "$OP"   # 期望最终 AUDIT_CLEAN
bash $OP/scripts/gates/convention_audit.sh "$OP"  # 期望最终 CONVENTION_CLEAN（含 §4.4/§4.2 检查）
bash $OP/scripts/gates/run_gate_all.sh    "$OP"   # L0/L1/L2/L4
PROCS=20 PAR=4 DEV=<空闲卡> TAG=t3c bash $OP/scripts/gates/race_soak.sh "$OP"   # 碰同步/落点时必须
python3 $OP/scripts/gates/arch_view_diff.py --macros "__CCE_AICORE__=310" <old.cpp> <new.cpp>   # 拆文件等价性
git -C <repo> status --short && git -C <repo> diff --check      # §8 最后两条
```

## 5. 边界与风险

1. **不改数值行为**：T2/T3 期望 `.o` 逐字节相同；T3b/T3c/T4/T5 允许 `.o` 变，但必须 L1 位级一致 + 全门禁。
2. **T3c 有一处"规范 vs 收益"的取舍**：§4.4 要求 arch35 的 AIV 用 `__simd_vf__` VF 融合，我们现在是普通向量指令。建议先只做"函数移出类 + 拆 Stage + 三张表"，把 VF 化单列一轮（有收益才做；无收益则在 `design.md` 记平台差异）。
3. **T4 的 TilingKey 模板化会改派发方式**：必须保证 gateMode×dtype 组合与现有一一对应，host 侧用 `static_assert` 钉档位取值，并用"新旧同 case 位级一致"验证。
4. **T3 拆文件时只搬不改**：一次只搬不顺手优化；用 `arch_view_diff.py` 证明展开后等价。
5. **`clang-format` 单独一个 commit**、只对改动文件跑，避免整文件重排掩盖真实改动。
6. **A3(910_93) 只能构建级验证**；T4/T8 至少保证编得过 + 出 whl。
7. **不与当前性能迭代抢道**：建议性能冻结后执行；若必须并行，只做 T1（纯删除/新增）与 T2（纯注释）。

## 6. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| `docs/switch_remediation_plan.md` | 本计划 **T4 的明细**（开关清单、白名单、`static_assert` 清单） |
| `docs/race_status_20260929.md` | soak 协议与竞态结论；T3c/T4/T5 的 soak 判据引用它 |
| `docs/validation.md §13+` | 各轮实测；T6 重构成 `docs/perf/round_NNN/`，validation 保留为总账 |
| `docs/a2_opt_status.md` | 内容已过时，T1 顺手更正 |
