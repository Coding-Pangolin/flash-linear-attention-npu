# Stable ABI 全量适配方案（唯一后端 · 对齐 vllm-ascend 性能 · 全场景覆盖）

> 分支：`feat/stable-abi-thin`
> 逐轮实测与排查过程见 [stable-abi-migration.md](./stable-abi-migration.md)
> 本文是**执行版**：一个算子要改哪里、什么算达标、场景怎么记录、还差什么。

## 0. 与上一版方案的差异（2026-09-11 实测后刷新）

| 项 | v1（方案刚写下时） | v2（本文，A1/A2 已落地） |
| --- | --- | --- |
| 覆盖 | 16/26，剩 10 个卡在 `alloc`/`helpers` 的 ATen 惯用法 | **25/26**：门面接线后 23 个由 codegen 生成 + 2 个手写；只剩 `npu_causal_conv1d`（等上游 #390 ABI） |
| 一次性通过 | 无 | `tests/regression_stable_full.py`（`_thin` 整体改道 stable）**243 PASS / 0 FAIL**，22 个算子被真实调用 |
| 性能 | 只有 GDR/KDA 两个数 | GDR 1.26×（**未达标**）、KDA 1.11×（达标）、`fast_gelu` 0.34×；stable 直连 0.0629–0.0757 ms，已在 vllm custom（0.073 ms）量级 |
| 后端取舍 | stable 可选、pybind 默认 | **stable 为唯一后端**；pybind 仅作过渡期 A/B 对照，随后整体删除 |
| 场景覆盖 | 口头"没丢场景" | 机器化：spec `scenarios` + `tools/stable_coverage.py` + parity 基线 JSON + 回退计数（Phase C，本轮新增设计） |

## 1. 三条需求对应到可测门禁

需求原话拆开是三条，每条都要有能自动跑的判据：

| 需求 | 含义 | 门禁（可自动判定） |
| --- | --- | --- |
| "性能对齐 vllm-ascend 量级" | 我们的 host 时间与 vllm-ascend 自研 custom op 同量级 | **stable 直连 host P50 ≤ 0.073 ms**（vllm custom 实测值） |
| "不能回退 thin" | ① 不靠回退到 pybind/ctypes 才跑得起来；② 相对 thin 不引入性能倒退 | ① 合法域内**零回退**（`stable_coverage.py`）；② `stable_public ≤ 1.15 × pybind_public` |
| "全部场景都记录覆盖走到 stable" | 每个算子的合法输入域逐轴声明，逐场景有 parity 记录，diff/回退能报警 | spec `scenarios` + `tests/stable_scenarios.json` 基线 + `FLA_NPU_THIN_TRACE=1` 回退计数 |

补充两条不变量：

- **正确性**：每场景 ctypes vs stable 逐输出 `diff == 0.0`（含 None 掩码与返回顺序），inplace / version 契约一致。
- **依赖**：wheel `py3-none-any`，包内只有 `libfla_npu_thin.so`；`nm -D` 里 0 个 `_ZN2at/_ZN3c10`；同一产物在 torch 2.7.1 与 2.9 上都能加载并通过 parity。
- **失败语义**：合法输入逐位一致；非法输入保证**报错、不崩**，报错类型不保证同型（与 vllm-ascend 一致）。

当前证据（221 / 910B3，本轮复跑）：

```
libfla_npu_thin.so  243 536 B
undefined  _ZN2at/_ZN3c10 = 0     aoti_torch_* = 41
regression_stable_full.py:  243 PASS / 0 FAIL,  "ALL PASS: full stable parity"
```

## 2. 算子清单与当前状态（26 = ctypes 全量）

"适配来源"只有两种：`GEN`（spec 驱动 codegen 生成）与 `手写`（复合语义，spec 表达不了）。
"全量演练"= 本轮 `regression_stable_full.py` 是否真实调用过它。

| # | 算子 | 适配来源 | 张量参数 | 全量演练 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | `npu_causal_conv1d` | **缺** | – | ✗ | 待 A3；`conv_states` 原地更新，等上游 #390 ABI 定稿 |
| 2 | `npu_causal_conv1d_bwd` | GEN | 10 | ✓ | `input_layout` 走 enum |
| 3 | `npu_chunk_bwd_dqkwg` | GEN | 14 | ✓ | 可选输出 `when` 条件 |
| 4 | `npu_chunk_bwd_dv_local` | GEN | 7 | ✓ | |
| 5 | `npu_chunk_fwd_h` | GEN | 9 | ✓ | |
| 6 | `npu_chunk_fwd_o` | GEN | 6 | ✓ | `output_layout` 走 enum |
| 7 | `npu_chunk_gated_delta_rule_bwd_dhu` | GEN | 12 | ✓ | `Tensor?` 槽 nullopt 化的第一个现场 |
| 8 | `npu_chunk_gated_delta_rule_bwd_finalize` | GEN | 19 | **✗** | 已注册，测试场景待补（Phase C5） |
| 9 | `npu_chunk_gated_delta_rule_fwd` | GEN | 18 | ✓ | 4 layout × flag 组合 |
| 10 | `npu_chunk_gated_delta_rule_fwd_h` | GEN | 9 | ✓ | |
| 11 | `npu_chunk_gated_delta_rule_fwd_prepare` | GEN | 16 | **✗** | 已注册，测试场景待补（Phase C5） |
| 12 | `npu_chunk_kda_bwd` | GEN | 26 | ✓ | 参数量最大 |
| 13 | `npu_chunk_kda_bwd_intra` | GEN | 14 | ✓ | |
| 14 | `npu_chunk_kda_fwd` | GEN | 19 | ✓ | 201 组 flag/layout 组合，本轮全绿 |
| 15 | `npu_chunk_local_cumsum` | GEN | 2 | ✓ | `output_dtype` 走 enum |
| 16 | `npu_chunk_scaled_dot_kkt` | GEN | 4 | ✓ | |
| 17 | `npu_fast_gelu_custom` | GEN | 2 | ✓ | 参数名是 `self`，不能按"方法接收者"过滤 |
| 18 | `npu_fast_gelu_custom_backward` | GEN | 3 | ✓ | |
| 19 | `npu_kda_gate_cumsum` | GEN | 4 | ✓ | |
| 20 | `npu_prepare_wy_repr_bwd` | GEN | 11 | ✓ | |
| 21 | `npu_prepare_wy_repr_bwd_da` | GEN | 8 | ✓ | |
| 22 | `npu_prepare_wy_repr_bwd_full` | GEN | 12 | ✓ | |
| 23 | `npu_recompute_w_u_fwd` | GEN | 8 | ✓ | |
| 24 | `npu_recurrent_gated_delta_rule` | 手写 | 11 | ✓ | |
| 25 | `npu_recurrent_kda` | 手写 | 13 | **✗** | 单独驱动 `regression_stable_abi_kda.py` 全绿，未并进全量脚本 |
| 26 | `npu_solve_tri` | GEN | 2 | ✓ | `layout` 走 enum；`ntd` 是上游内核坏域，见 §8 |

统计：**生成 23 + 手写 2 = 25/26 有 stable 适配**；缺口 4 处 =
1 个未适配（`npu_causal_conv1d`）+ 3 个已适配但未纳入全量演练（8、11、25）。
`tools/op_stable_codegen.py --parse-only` 会逐算子打印这份状态，可直接进 CI。

## 3. 新增一个算子的完整改动面

目标是"**只加一个 JSON**"。现状已基本达到，只有复合语义的算子需要手写。

### 3.1 标准流程（绝大多数算子）

```
1) op_specs/aclnn_<op>.json          # 唯一手写输入
2) python tools/op_stable_codegen.py --all
     -> csrc_stable/generated/ops_stable_generated.inc      (C++ 适配 + STABLE_TORCH_LIBRARY 注册)
     -> fla_npu/ops/ascendc/_stable_generated.py            (Python 包装，签名与 ctypes 逐参数对齐)
3) python csrc_stable/build_stable.py --no-debug-probe --out libfla_npu_thin.so
4) tests/regression_thin_ops.py 里加一个场景（同一份场景同时跑 ctypes 与 stable）
```

改动计数：

- **spec JSON 1 个**（`aclnn_name` / `python_name` / `args` / `outputs` / `python` / `scenarios`）；
- **测试场景 1 个**；
- **白名单 0 处**——`_get_direct_op` 按算子名动态解析；`MUTATED_ARGUMENTS` / `MUTATION_FLAGS` 只在"有 inplace 语义"时才加一行；
- 生成产物 2 个（`.inc` 与 `_stable_generated.py`）由脚本产出。

spec 里三类"坑"已有固定写法，新算子照抄即可：

| 参数形态 | 写法 | 例 |
| --- | --- | --- |
| `int[]`（`cu_seqlens` / `chunk_indices`） | 表示为 host int64 张量，C++ 用 `AclIntArrayView` 读值 | 12+ 个算子 |
| `char_ptr`（layout / output_dtype） | spec 加 `"enum": ["BSND","BNSD","TND","NTD"]`，生成器出 int 码 ↔ 字符串映射 | `solve_tri`、`chunk_fwd_o` |
| 输出 `alloc` 用 ATen 惯用法 | **照抄 ATen 原文**，由门面 `at_shim` 承接，不改 spec | 8 个算子 |

### 3.2 手写适配的触发条件（目前只有 2 个）

满足任一条才手写，否则一律走生成：

1. **输出需要是调用方张量本身**（inplace 返回，stable 侧二次接管会双释放）；
2. **需要按 layout 字符串分支**决定 descriptor 形状；
3. **需要在 Python 侧构造 scratch 张量**（如 `inplace_final_state=False` 的 KDA）。

手写件位置：`csrc_stable/src/stable_recurrent_gdr.cpp`、`stable_recurrent_kda.cpp`；
注册在唯一的 `STABLE_TORCH_LIBRARY(_IMPL)` 块（`stable_ops.cpp`）。

## 4. 性能方案

### 4.1 成本模型（实测拟合）

```
stable_host ≈ 4.3 µs (dispatcher) + ~2 µs × 张量参数个数 + 该 kernel 固有 host tiling
ctypes_host ≈ 0.24–0.63 ms   ← 其中 91% 是 Python 里建销 descriptor 与三段式 FFI
```

所以两类算子的表现完全不同：

| 算子 | 对照（P50, ms） | stable（P50, ms） | 比值 | 判定 |
| --- | --- | --- | --- | --- |
| `fast_gelu_custom`（2 张量参数） | ctypes 0.2382 | 0.0802 | **0.34×** | 远超 ctypes |
| `recurrent_kda`（13 张量参数，wrapper 已较重） | pybind 0.0833 / 0.0842 | 0.0923 / 0.0938 | **1.108 / 1.113×** | 达标（≤1.15×） |
| `recurrent_gated_delta_rule`（11 张量参数，wrapper 极轻） | pybind 0.0734 / 0.0738 | 0.0932 / 0.0933 | **1.264 / 1.270×** | **未达标**，见 B2/B4 |
| 全量算子 stable **直连** | vllm-ascend custom 0.073 | 0.0629–0.0757 | ≈1.0× | 已在同一量级 |

关键读法：stable 的**裸调用**已经不输 vllm custom；GDR 的 1.26× 全部来自
"我们的 Python 公共层（stream 查询 + mutation 契约 + 逐参数装箱）在 pybind 上更便宜"，
不是 dispatcher 本身（裸 dispatcher 只有 4.3 µs）。

### 4.2 优化项与顺序

| 编号 | 内容 | 状态 | 预期 |
| --- | --- | --- | --- |
| B1 | 逐参数解包：`fill_meta(handle)` 不构造 Tensor、`empty_strided` 直分配、`torch.ops` 句柄缓存 | 已完成 | 基线 |
| B2 | `int[]` 按 `tuple(values)` 缓存 host int64 张量 | 待做 | 省 3–8 µs/次（decode 长度序列高度重复，命中率高） |
| B3 | stream：每调用 raw accessor（~1.2 µs），**不做进程级缓存** | 已完成（含 vLLM 崩溃教训） | 正确性优先 |
| B4 | 校验分层：schema 免费 + C++ 廉价断言 + 算子自身合法域；全量校验只在 `FLA_NPU_THIN_VALIDATE=1` | 待做 | 省 30–50 µs/次（若误搬 ctypes 校验则倒亏） |
| B5 | 每算子交替采样 A/B（两轮，P50+P90） | 待做（当前只有 3 个算子） | 出 26 行验收表 |

**"不能回退 thin" 的落点**就是 B2/B4：GDR 现在 1.26×，必须压回 ≤1.15×；
压不下去就把"已定位到 dispatcher 逐参数成本"的书面结论记进去，而不是悄悄换回 pybind。

### 4.3 与 vllm-ascend 的内部分工差异（口径对齐）

vllm-ascend 走的是同一条 dispatcher 路线（C++ op + `EXEC_NPU_CMD`），差异只在：

1. 它把校验分摊到 **schema（免费）+ C++ 少量 `TORCH_CHECK` + 算子 `CHECK_COND`**，热路径不做 Python 全量校验；
2. 它的 wrapper 不做 mutation 契约（`state` in-place 后版本号不动），我们做了——这是我们多出来的固定成本，属**有意保留**的正确性差异。

## 5. 场景覆盖记录（Phase C，"全部场景都记录覆盖"）

现状：`regression_stable_full.py` 的 243 PASS 是**事实上的**覆盖，但还不是**可查询、可报警**的。补齐 C1–C5：

### C1. spec 增 `scenarios`（合法域契约，机器可读）

```json
"scenarios": {
  "layout": ["BSND", "BNSD", "TND", "NTD"],
  "varlen": [false, true],
  "flags":  ["output_final_state", "disable_recompute"]
}
```

平面的"轴 → 合法值"映射，而不是用例对象列表——diff 里读得出来，也便于与
C2 从 spec 反推出的轴逐条对账。**维度的合法值必须从 ctypes 实现反推，不允许凭印象写**；
反推不出来的轴（例如没有 `enum` 表的 `char_ptr`）在矩阵里标 `UNVERIFIABLE`，而不是假装已声明。

### C2. `tools/stable_coverage.py`（已实现，离线门禁，不需要 NPU）

```
python tools/stable_coverage.py           # 覆盖矩阵；已登记缺口 = KNOWN GAP，其余 FAIL
python tools/stable_coverage.py --axes    # 逐算子列出反推出的轴与合法值
python tools/stable_coverage.py --strict  # 不认 baseline，任何缺口都 FAIL（发版前用）
python tools/stable_coverage.py --json    # 机器可读，供 CI 消费
```

三项判定，全部来自仓库内的文件：

1. **适配器覆盖**：ctypes 暴露的每个 `npu_*` 都要有 stable 适配器（`.inc` 里的
   `kSchema_<op>` 或 `_stable.py` 里的同名函数）。缺失 = FAIL，除非写进
   `tests/stable_coverage_baseline.json` 并给出理由与移除条件——**缺口是登记制，不是容忍制**。
2. **场景轴声明**：从 spec 反推轴（`char_ptr` 的 `enum` → layout/dtype 轴及合法值；
   `cu_seqlens`/`chunk_indices`/`actual_seq_lengths`/`query_start_loc` → varlen 轴；
   `cache_indices`/`num_accepted_tokens` → spec-decode 轴；布尔入参与输出 `return_when` → flag 轴），
   再拿 `scenarios` 逐条对账；声明了推导不出来的轴、或声明了 spec 表达不了的值 = FAIL。
3. **spec 与注册一致性**：注册了却缺 spec、有 spec 却没注册、生成了却没有 Python 包装 = FAIL。

当前输出（221/910B3 时代的仓库状态，本地离线可复现）：

```
ctypes operators: 26
stable adapters : 23 generated + 2 hand-written
npu_causal_conv1d   -   0 axes   KNOWN GAP  (waiting on #390)
... 其余 25 行 OK ...
ALL COVERED: every ctypes operator has a stable adapter (recorded gaps only)
```

`--strict` 退出码 1 并列出 `npu_causal_conv1d: no stable adapter`——即 A3 完成后
baseline 必须清空，否则发版门禁不放行。

### C3. parity 基线入库

由 `scenarios` 生成用例，执行后把每场景的 `diff` 写进 `tests/stable_scenarios.json`。
任何一次改动导致场景丢失或数值变化，都会在 diff 里显形。

### C4. 运行期回退可视化

`FLA_NPU_THIN_TRACE=1` 时打印/计数"算子 + 场景标签"；CI 在合法域内断言计数为 0。

### C5. 补全本轮暴露的 3 个演练缺口

`npu_chunk_gated_delta_rule_bwd_finalize`、`npu_chunk_gated_delta_rule_fwd_prepare`、
`npu_recurrent_kda` 已适配但没进全量脚本——先补场景，避免"注册了=覆盖了"的错觉。

## 6. 剩余工作与推进顺序

| 阶段 | 内容 | 前置 |
| --- | --- | --- |
| A3 | `npu_causal_conv1d` + `causal_conv1d_update` 适配（`conv_states` in-place、`int[]`、layout enum） | 上游 #390 ABI 定稿 |
| C1–C2 | `scenarios` + 覆盖矩阵门禁 | **C2 已实现**；C1 的逐算子轴值核对待补 |
| C5 | 补 3 个演练缺口 | C1 |
| B2/B4 | int 缓存 + 校验分层，把 GDR 压回 ≤1.15× | 无 |
| B5/C3 | 26 算子 A/B 表 + parity 基线入库 | B2/B4、C1 |
| D1 | 默认后端链改为 stable 单条（去掉 `_get_thin_op` 的默认分支），`FLA_NPU_THIN_ABI=pybind` 仅留给 A/B | C1–C5 全绿 |
| D2 | 删 `_C_thin`：从 `setup.py` / `scripts/build_wheel.py` 移除构建与打包路径 | D1 |
| D3 | 发布矩阵：`py3-none-any`、`torch>=2.7.1` 下限（运行期 `aoti_torch_abi_version()` + 符号检查）、SOC 分 OPP 包 | D2 |

建议顺序：**C1 → C2 → C5 →（A3 就绪后）→ B2 → B4 → B5/C3 → D1 → D2 → D3**。
把门禁（C1/C2）放在最前，后续每个算子的接入自动被记录，避免"迁移完才发现没登记"。

## 7. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| dispatcher 逐参数成本（~2 µs/张量）是硬下限 | 11 张量参数以上算子难做到 1.0× | 主验收口径用"**stable 直连 vs vllm 量级线（0.073 ms）**"，而不是"vs pybind"；pybind 的公共层开销本就不该被当作基线 |
| 轻 wrapper 算子比值偏高（GDR 1.26×） | 30 次/step × +0.02 ms ≈ +0.6 ms/step | B2/B4；并把该比值写进文档而非隐藏 |
| 删除 `_C_thin` 后回退手段变少 | 出问题只能退 ctypes（更慢）或回滚版本 | D1 与 D2 分成两个 commit，中间留一个"stable 为默认但 pybind 仍可显式启用"的可回退点 |
| 上游内核坏域（`solve_tri` 的 `ntd`） | 无法"覆盖" | 记录为内核问题；stable 与 ctypes 行为一致，不计入覆盖缺口 |
| 非法输入报错类型不同型 | 客户代码若 `except` 具体异常会受影响 | 文档写明契约；`FLA_NPU_THIN_VALIDATE=1` 可开全量校验换取同型报错 |
| NPU 内部 format 张量（非 ND） | stable 无法读取 format | 现网统一 ND；若需支持，Python 侧查 `npu_get_format` 把 code 传进 descriptor |
| `stable_abi_audit.py` 只在本地跑 | 新引入的 ATen 依赖可能悄悄回流 | 把 audit 挂到 CI（检查 0 个 `_ZN2at/_ZN3c10`、wheel tag、`.so` 单一） |

## 8. 已知限制（记录，不是缺口）

- `solve_tri` 的 `ntd`/转置域：上游内核自身返回全 0，ctypes 亦然——两路径一致，记为内核限制。
- NPU 内部 format（非 ND）张量：stable 侧无稳定 API 可读 format，当前统一 ND。
- 版本下限：头文件下限 torch 2.9，运行期符号下限 **2.7.1**（无 debug probe 构建）。
- `torch.ops.fla_npu_thin.*` 是**实现细节**，对外只暴露 `fla_npu.ops.ascendc.*`。
