# 开关整改计划：把"实验矩阵"收敛成"两套 arch 实现 + 一张布局表 + 一个 debug 构建"

> 2026-09-29，基于 `2cc812a`（R20，本文件与 `op_kernel/pre_process_fwd_kernel_merged.cpp`
> 在该 commit 均无改动）。目的：上库前把 `PPFM_*` 实验开关清理掉，**保留已实测最优的路径**，
> 同时**不改变任何一条已验收的数值行为**（判据：每个批次都过 L1 位级 + L2/L4 门禁；
> 碰同步/落点的批次加跑进程级 soak）。
>
> 读之前请先明确一条原则：**架构分档用的 `#if` 保留**（A2/A3 确实没有 L0C→UB 通道，
> 代码必须不同），要删的是"**同一个 arch 内还留着的多条可选路径**"。

## 1. 现状盘点（数据）

`op_kernel/pre_process_fwd_kernel_merged.cpp`：

| 指标 | 值 |
|---|---|
| 文件总行数 | **2335** |
| 预处理指令行 | **413**，其中 `#if/#ifdef/#ifndef` 指令 **142** 条 |
| 处于条件编译块内的代码行 | **1028（44.0%）**，其中处于 `PPFM_*` 条件内的 **965（41.3%）** |
| 可被命令行覆盖的实验开关 | **20 个** = 19 个开关名（21 个 `#ifndef PPFM_X / #define PPFM_X 默认值` 块；`PPFM_VTMP_UB` 有 3 处，其中 2 处是同一分支的重复）+ `PPFM_DEBUG_HEADER`（无默认块，只在手工 `-D` 时生效） |
| `static_assert` | **0** |
| 内核读环境变量 | 无 ✅ |
| host 读环境变量 | **`PPFM_FORCE_COLSPLIT`**（测试钩子，留在生产 host 代码里） |
| 死文件 | `op_kernel/*_v1_vector.cpp.txt`(26 KB)、`*_v2_cube.cpp.txt`(45 KB)、`*.cpp.bak`(19 KB)、`op_host/CMakeLists.txt.wip` — 共 260+ 处 `PPFM_` 噪音 |

### 1.1 开关清单与分类

**A 类：已定稿的最优路径（默认就是最优，0 分支已被实测否掉）→ 删开关、留路径**

| 开关 | 默认 | 0 分支的身份 | 证据 |
|---|---|---|---|
| `PPFM_KDA_DECAY_VEC` | 1 | 逐点 `SetValue+Exp2Scalar`（每 chunk 256 次全栅栏） | §15，位级一致，b −46.8% |
| `PPFM_T1_FIXPIPE_BF16` | 1 | AIV 读回 t1F_→Cast→写回 | §17，位级一致，−4.7%/−4.1% |
| `PPFM_AIV_EVENTS` | 1 | `PipeBarrier<PIPE_ALL>`（P1a 前） | §14，−13.7%/−7.7% |
| `PPFM_LEGACY_PROBE_READS` | 0 | 4 处过渡探读（纯开销） | §20，位级一致 + soak 0/20，−3.2~4.2% |
| `PPFM_TILE_MMAD` | 1 | CATLASS `BlockMmad` 调用点（A1 之前的路径） | tile 路径已是生产路径，且 BlockMmad 版**只是 4 行调用点**（`RunMmadNT/TA` 本体无条件编译、当前无人调用） |

**B 类：架构差异（不是"选择"，是"必须不同"）→ 保留但改为 arch 策略，取消独立开关**

| 开关 | 950 | A2/A3 | 说明 |
|---|---|---|---|
| `PPFM_VTMP_UB` | 1 | 0 | mm1 的 C：L0C→UB vs 经 GM |
| `PPFM_DH_CV` | 1 | 0 | mm2 的 C 同上 |
| `PPFM_T2_CV` | 1（依赖 DH_CV） | 0 | mm4 的 C 同上 |
| `PPFM_H_UB` | 1 | 0（被联动关掉） | h 常驻 UB（依赖 dH 已在 UB） |
| `PPFM_M_UB` | 1（依赖 H_UB&&DH_CV） | 0 | m 常驻 UB |
| `PPFM_UB_SHARE` / `PPFM_NSLOT` | 1 / 1 | 1 / 1 ⚠️ | 依据是 950 的"每子核独立 UB bank"语义 ⇒ **A2 上是否成立必须先确认**（见 §5.1） |
| `PPFM_LEGACY_CACHEOPS` | 0 | 1 | A2 必须手工 DCCI/DSB（chunk0 的 `h≡0` 探针实测 1.6e-2）⇒ 属**正确性必需** |
| `PPFM_ARCH_IS_950` / `PPFM_XCORE_MODE` | 1 / 0x4 | 0 / 0x2 | 保留，作为**唯一的 arch 谓词** |

**C 类：诊断/探针（必须离开生产二进制）→ 删除或收进独立 debug 头**

`PPFM_DIAG`(45 行)、`PPFM_DIAG_CHUNKS`、`PPFM_DEBUG_HEADER`(57 行，只在手工 `-D` 时生效，**生产里永远是死代码**)、
`PPFM_RD_PROBE`(50 行)、`PPFM_VTMP_UB_DIAG`(22 行)、`PPFM_SENTINEL_PROBE`(7 行)。

> ⚠️ 其中 `PPFM_DIAG` / `PPFM_RD_PROBE` / `PPFM_DEBUG_HEADER` 会**往 `hm`（算子真实输出）里写诊断数据**
> （m 半边第 0 行 / 第 0..N 行，靠"验收时排除这些行"绕过）。上库版本**不能保留这类通路**。

**D 类：测试钩子 → 移出生产代码**

- host `std::getenv("PPFM_FORCE_COLSPLIT")` ⇒ 改成 tiling 里的隐藏字段，或只在测试构建里编入。

**E 类：布局常量（保留，但要"一张表 + 断言"，不能被当成开关）**

`PPFM_SUB / PPFM_SEG / PPFM_RB / PPFM_SBRB / PPFM_SEGROWS / PPFM_NSLOT / PPFM_CORE_WS_BYTES /
PPFM_VEC_UB_BYTES` 与 `UB_*` 偏移；`struct.h` 里的 `PPFM_CORE_M_OFFSET / T1 / T2 / MNEXT`。
它们目前**没有任何编译期校验**：UB 预算超了要等运行时才炸，host `CORE_WS_BYTES` 与 kernel 偏移靠人眼对齐。

**F 类：命名**：`PPFM_K/V/CHUNK`（aclnn 里的形状常量）、`PPFM_GATE_USE_* / PPFM_DTYPE_*`（tiling 枚举）不是开关，
但"PPFM_"前缀让人以为都是实验开关 ⇒ 建议改为 `kDimK` / `GateMode::G` 这类名字。

## 2. 目标形态

```
op_kernel/
  pre_process_fwd_kernel_merged.cpp     # 正文只有一份；arch 差异用 if constexpr / 模板 + 一个小策略结构
  pre_process_fwd_kernel_merged_arch.h  # 唯一允许 #if __CCE_AICORE__==310 的地方：UB 布局表 + 能力开关
  pre_process_fwd_kernel_merged_debug.h # 可选：全部诊断（默认不参与编译）
```

三条硬指标：

1. **kernel 里 `PPFM_*` 只剩白名单**：`PPFM_ARCH_IS_950`（arch 谓词）与布局/枚举常量；
2. **`#if` 数量从 142 降到 ≤ 25**（arch 分档 + debug 头），条件内代码行从 916 降到 ≤ ~250；
3. **`static_assert` 覆盖**：UB 字节 ≤ `ArchTag::UB_SIZE`、workspace 偏移单调不重叠、host/kernel 的
   `CORE_WS_BYTES` 一致、`PPFM_SUB==2` 与子核数一致。

## 3. 分批落地（每批一个 commit，可独立回退）

| 批 | 内容 | 预期净效果 | 验证 |
|---|---|---|---|
| **S0** | 删死文件（`*.txt`/`*.bak`/`CMakeLists.txt.wip`）；删 `PPFM_VTMP_UB` 的**第二处重复定义**（第 309 行那份永远不会生效，因为第 70 行先定义了） | −90 KB 死文件、少一处自相矛盾的默认值 | 编译 + L1 位级（应逐字节相同） |
| **S1** | 删 C 类诊断（D 类钩子同时移出）；`gate_l0.py` 里"必须 `PPFM_DIAG == 0`"的检查改成"源码不得含 `PPFM_DIAG`" | −180 行，且**算子输出不再有诊断写入通路** | 编译 + L1 位级 + L2 + L4 |
| **S2** | 删 A 类（`KDA_DECAY_VEC`/`T1_FIXPIPE_BF16`/`AIV_EVENTS`/`LEGACY_PROBE_READS`/`TILE_MMAD`）的 0 分支与开关；连带删 `RunMmadNT/RunMmadTA` 与 `MmBlockNT/TA` 类型别名 | −80~120 行、少 5 个开关 | 编译 + L1 位级（`.o` 应逐字节相同）+ L2 + L4 |
| **S3** | B 类收敛：新增 `arch.h` 的策略结构（`kUseCV`/`kStateInUb`/`kNeedDcci`/`kWaitPipe`/`kMIdentityBuild`…），正文改用 `if constexpr` 取用；删掉 `VTMP_UB/DH_CV/T2_CV/H_UB/M_UB/LEGACY_CACHEOPS` 这 6 个开关名，`UB_SHARE/NSLOT` 按 §5.1 的结论改成 arch 常量（不再是开关） | −0 行但**开关数 20 → ~8**（含 `H_UB_A2` 的去向，见 §5.2） | 编译（950+A2）+ L1 位级 + L2 + L4 + **950/A2 各跑一次 soak 0/20** |
| **S4** | 布局表 + 断言：把 `UB_*` 偏移与 `PPFM_*_UB_BYTES` 集中到 `arch.h`，加 §2 的四组 `static_assert`；host 侧加"workspace 字节数一致"的编译期/单测校验 | 把"靠人眼"变成"编译器管" | 编译（故意改错一个偏移应编译失败，作为断言自检）+ 全门禁 |
| **S5** | 收尾：`PPFM_FORCE_COLSPLIT` 移出生产；aclnn/枚举改名；`docs/` 里补"最终形态说明"并归档本计划 | 代码里不再有实验痕迹 | 全门禁 + 两平台 soak |

> 顺序理由：先删"绝对死"的东西（S0/S1）→ 再删"已定稿"的 0 分支（S2，风险最低，`-D` 一开就能回退）
> → 才动架构策略（S3，风险最高，但此时噪音已经清干净，diff 好读）→ 最后上断言与收尾。

## 4. 不影响正确性的保障

1. **每批都过 L1 位级**：整改是"搬运代码 + 删死分支"，**期望 `.o` 逐字节相同**（S2 之后应完全相同）。
   唯一允许不同的批次是 S4（加了断言，不改代码路径）。
2. **保留等价回退**：S2/S3 的每一步在提交信息里写清"回退方式 = `git revert`"；
   `PPFM_LEGACY_PROBE_READS` / `LEGACY_CACHEOPS` 这类**曾经掩盖过竞态**的开关，若删，必须
   在同一 commit 的 message 里留一句"历史上它是为了掩盖 X，现已由结构性修复替代"。
3. **碰同步/落点的批次（S3）必须加跑进程级 soak**：`PROCS=20 PAR=4`（见 `docs/race_status_20260929.md`，
   单进程复跑对这类竞态不敏感）。两平台都要跑：950 在 247/246，A2 在 221。
4. **A3（910_93）只能构建级验证**：S3/S4 必须在 def.cpp 的 `ascend910_93` 上至少"编得过 + 出 whl"。

## 5. 必须先确认的语义问题（否则会把错误固化进代码）

### 5.1 `PPFM_UB_SHARE=1` 在 A2 上是否成立？（最高优先级）

它的依据（源码注释 R0.8）是 **950 的 `SPLIT_M` 语义**："两个 AIV 子核各自 bank 的同一偏移"。
但 `PPFM_UB_SHARE` 目前是**全局 1**，于是 A2 也 `PPFM_NSLOT=1`——而更早的注释明确写过
"任何两个子核都会写的 scratch 必须按 `subIdx_` 切成两份，否则会出现跨子核数据竞争"。
两者是相反的说法。**A2 上没有 L0C→UB / SPLIT_M**，所以它是否也有"每子核独立 bank"必须实测确认：

```text
实验：A2(221) 上把 PPFM_UB_SHARE 强制 0/1 各跑一遍 ——
      L1 位级 + L2 + L4 + 同一 kernel 连跑 6 次 dump 比对（看是否有 §28 那种"全对但非确定"）。
判据：若 0/1 都位级一致且都确定 → bank 语义在 A2 也成立，可以统一；
      若 1 出现抖动 → A2 必须 NSLOT=2，S3 里按 arch 给 NSLOT，而不能一刀切。
```

### 5.2 `PPFM_H_UB_A2=0` 的原因要不要在本轮收口

注释（R9）写的是：910B 上"数值全对但**非确定**"（同 kernel 连跑 4 次有 2/3 次差异），
疑似把 A2"T2 走 GM"那条边的残余窗口顶到表面，因此默认关。
⇒ 整改时两条路选一条，**不要留着"默认关的半个实现"**：

- (a) 先按 R6 同款把 A2 的 `T2` 边收口（或补上必要的 DCCI/wait），再打开；
- (b) 明确"本轮不做 A2 常驻状态"，**删掉 `H_UB_A2` 的实现**，只在文档里留理由。

### 5.3 默认值就是行为契约

所有 A 类开关的默认值都已过门禁，但**"默认值被验证过"≠"非默认值可以随便删"**：
删之前确认该开关**没有任何脚本/CMake 传 `-D`**（现已确认 build 文件里没有），
且 `gate_l0.py`（它会 grep `PPFM_TILE_MMAD` / `PPFM_DIAG` 的定义）同步改过。

## 6. 验收标准（可机械检查）

```bash
OP=<算子目录>
# 1) 开关白名单：只允许 arch 谓词 + 布局常量
grep -o 'PPFM_[A-Z0-9_]*' $OP/op_kernel/pre_process_fwd_kernel_merged.cpp | sort -u
#    期望：仅 PPFM_ARCH_IS_950 / PPFM_XCORE_MODE / PPFM_SUB / PPFM_SEG / PPFM_RB /
#          PPFM_SBRB / PPFM_SEGROWS / PPFM_NSLOT / PPFM_CORE_WS_BYTES / PPFM_VEC_UB_BYTES / UB_*
# 2) 不再有实验模式
grep -rn 'getenv' $OP/op_host $OP/op_kernel        # 期望：空
grep -rn 'PPFM_DIAG\|PPFM_RD_PROBE\|PPFM_DEBUG_HEADER\|PPFM_SENTINEL_PROBE\|PPFM_VTMP_UB_DIAG' \
     $OP/op_kernel                                  # 期望：空
ls $OP/op_kernel/*.txt $OP/op_kernel/*.bak $OP/op_host/*.wip   # 期望：不存在
# 3) 规模
grep -c '^\s*#' $OP/op_kernel/pre_process_fwd_kernel_merged.cpp   # 期望：条件编译点 ≤ 25
# 4) 门禁
bash $OP/scripts/gates/run_gate_all.sh $OP          # L0/L1/L2/L4 全绿
PROCS=20 PAR=4 DEV=<空闲卡> TAG=clean bash $OP/scripts/gates/race_soak.sh $OP   # SOAK_CLEAN
```

## 7. 与主线的关系

- 这份计划**不与当前性能迭代抢道**：它是"性能冻结之后的收尾"，每一批都可独立回退。
- 建议先做 S0/S1/S2（纯删除，风险最低、收益最直观：源码里立刻少 15 个开关），
  等 §5.1 的 A2 bank 语义确认、A2 的 h 常驻去向定了，再做 S3/S4。
- 完成 S0~S4 后，`docs/validation.md` 里那些"某开关=1 时 −x%"的记录仍然有效，
  但要补一句"该开关已删除，行为即默认路径"。
