# Stable ABI 全量适配方案（唯一后端，性能对齐 vllm-ascend）

> 目标分支：`feat/stable-abi-thin`
> 前置结论与实测见 [stable-abi-migration.md](./stable-abi-migration.md)
> 一句话目标：wheel 里只留 `libfla_npu_thin.so`（无 `_C_thin`、无 cpXXX、无
> ATen/c10 符号），26 个算子在 ctypes 支持的全部场景上由 stable 提供服务，
> 性能不低于 vllm-ascend 量级，且逐场景 parity 记录在案。

## 1. 验收口径（可测、可查）

| 维度 | 指标 | 证据形式 |
| --- | --- | --- |
| 覆盖 | ctypes 合法域内零回退：26 个算子 × 各自场景矩阵（layout × dtype × varlen × flags）全部由 stable 承接 | `tools/stable_coverage.py` 输出的覆盖矩阵 + `tests/stable_scenarios.json` 基线 |
| 正确性 | 每个场景 ctypes-vs-stable 逐输出 `diff == 0.0`（含 None 掩码与返回顺序）；mutation 契约（version/grad）一致 | 自动生成的 parity 驱动 + 结果 JSON；multi-stream 事件归属测试 |
| 性能 | 公共路径 host P50 ≤ 1.15× 现网 pybind 路径；且 stable 直连 ≤ vllm-ascend custom（实测 0.073 ms） | 交替采样 A/B（两轮一致 ±5%），P50/P90 双指标 |
| 依赖 | `WHEEL: Tag: py3-none-any` + 包内仅 `libfla_npu_thin.so`；该 .so 未定义符号里 0 个 `_ZN2at/_ZN3c10`；同一产物在 torch 2.7.1 与 2.9 上加载并通过 parity | `tools/stable_abi_audit.py`、跨版本加载矩阵 |
| 失败语义 | 合法输入逐位一致；非法输入保证报错、类型不保证同型（与 vllm-ascend 一致） | 文档契约 + 测试矩阵 |

## 2. 现状基线（2026-09-11）

- 覆盖：**16/26**（2 个手写 + 14 个 codegen 生成）；其中 6 个已实测 parity 0.0。
- 未覆盖 10 个：全部卡在同一类问题——输出/helpers 写成 ATen 惯用法；门面
  `csrc_stable/include/thin_stable/at_facade.h` 已实现，只差接线。
- 性能实测（221/910B3，交替采样 P50）：`recurrent_kda` stable/pybind = **1.11×**、
  `recurrent_gated_delta_rule` = **1.26×**、`fast_gelu` stable/ctypes = **0.34×**；
  stable 直连 0.0629–0.0757 ms，低于 vllm-ascend custom 的 0.073 ms。
- 依赖：已能产出 `py3-none-any` + 单个 `libfla_npu_thin.so` 的 wheel，且该产物在
  torch 2.7.1 / 2.9 上都跑通 parity。

## 3. Phase A：覆盖 16 → 26

### A1. 门面接线（解锁 8 个 alloc + 5 个 helpers 算子）

生成器改动（`tools/op_stable_codegen.py`）：

1. 对 `alloc` spec 额外暴露以参数名命名的 `at::Tensor` 视图（alloc 原文就是按参数名
   引用，如 `v.size(2)`），元数据仍走 `TensorMeta`；
2. `char_ptr` 参数在该 spec 内变成 `std::string`（`output_layout == "BNSD"` 这类比较
   需要字符串），由 enum 解码得到；
3. `alloc == "at::Tensor()"` 的输出槽按 `Tensor?` 处理并打包 `nullopt`；
   `when` 为假的槽同样必须打包 `nullopt`（不能塞未定义 Tensor——2.7.1 崩溃根因）；
4. spec `helpers` 去重后在聚合文件里只发一份（门面提供其所需的 `at::Tensor` /
   `at::empty` / `at::kFloat` 等子集）。

门禁：每个新生成算子在 910b 上跑一次 parity；至少覆盖 dense 一种布局 + 一个 flag 变体。

### A2. 两个复合 spec

`npu_chunk_gated_delta_rule_fwd`、`npu_chunk_kda_fwd`：既有 layout-aware helpers、
varlen chunk 数推导、又有 flag 驱动的可选输出。

- 复用现有 spec 的 `helpers`（C++ 语义）经门面编译；
- 沿用已在手写适配里验证过的三条规则：`data_ptr` 要减 `offset*itemsize`、storage
  extent 用 `get_storage_size`、条件输出打包 `nullopt`；
- 验收直接用既有矩阵：`chunk_kda_fwd` 的 4 layout × dense/varlen × 16 flag（192 组）
  + 9 个 shape/dtype 变体；composite 的 A2 21 场景 + A5 域。

### A3. conv1d 家族（等上游 #390 合入）

`npu_causal_conv1d`（legacy，BSH/BSND/TND 三种 layout + `conv_states` 原地更新）与
`causal_conv1d_update`（#390 ABI）。要点：

- `conv_states` 是 in-place ref → 复用 KDA 的处理（内核写回、Python 层返回调用方张量，
  `MUTATION_FLAGS` 已有 `conv_states` 条目）；
- `query_start_loc` / `cache_indices` / `num_accepted_tokens` 是 `int[]` → 已由
  “int 数组当 host 张量” 机制覆盖；
- 三种 layout 用 enum 编码（同 `solve_tri`）。

### A4. 收尾：把“上游内核坏域”与“覆盖缺口”分开

例如 `solve_tri` 的 ntd（ctypes 本身返回全 0、thin 也不是 tnd 转置）属于上游内核
问题：stable 与 ctypes 表现一致，记录为已知域限制，不计入覆盖缺口。

**Phase A 门禁**：26/26 注册 + 每个算子的场景矩阵在 ctypes 合法域内零回退。

## 4. Phase B：性能对齐 vllm-ascend 量级

已知成本模型（实测）：`stable_host ≈ 4.3µs(dispatcher) + ~2µs × 张量参数个数 +
该 kernel 固有 host tiling`；而 `ctypes_host ≈ 0.24–0.63 ms`（被 Python 里
descriptor 建销与三段式 FFI 主导）。因此：

1. 不再回退 thin 的落点：stable 直连已低于 vllm custom（0.063–0.076 vs 0.073）；
   公共路径多出的部分是我们自己的 Python 层（stream 查询 + mutation 契约），
   两个后端都要付。
2. B1 逐参数解包：已完成（`fill_meta(handle)` 不构造 Tensor、`empty_strided` 直接
   分配、`torch.ops` 句柄缓存）。
3. B2 `int[]` 缓存：按 `tuple(values)` 缓存 host int64 张量，varlen 热路径省
   3–8 µs/次（decode 场景长度序列高度重复，命中率高）。
4. B3 stream：维持每调用 raw accessor（~1.2 µs，多线程安全）；不做进程级缓存
   （vLLM 崩溃根因）。
5. B4 校验分层（照 vllm-ascend 的分工）：schema（免费）+ C++ 廉价检查（如 `scale`
   必填、dtype 断言）+ 算子自身合法域校验。禁止把 ctypes 的 Python 全量校验搬进
   热路径（+0.03–0.05 ms/次）；`FLA_NPU_THIN_VALIDATE=1` 时才启用全量校验。
6. B5 验收：每个算子交替采样 A/B（两轮，P50+P90），要求
   `stable_public ≤ 1.15 × pybind_public`，且 `stable_direct ≤ 0.073 ms`
   （现网 11 张量参数算子的 vllm 量级线）。

**Phase B 门禁**：26 个算子的 A/B 表全部达标，或对未达标者给出“已定位到 dispatcher
逐参数成本”的书面结论并记录为已知下限。

## 5. Phase C：场景记录与覆盖门禁（全部场景都记录覆盖）

### C1. spec 增加 `scenarios`（合法域契约）

```json
"scenarios": [
  {"layout": ["BSND","BNSD","TND","NTD"], "varlen": [false,true],
   "flags": {"output_final_state": [false,true], "disable_recompute": [false,true]}},
  {"kernel_broken": ["ntd"], "note": "ctypes 自身返回全 0，见 issue #xxx"}
]
```

它是“必须覆盖”的声明，也是生成测试用例与覆盖矩阵的输入。

### C2. `tools/stable_coverage.py`（离线门禁，不需要 NPU）

- 输入：所有 spec（含 `scenarios`）+ 已注册算子的清单（从 `_stable_generated` /
  `_stable` 的函数名推导）；
- 输出：覆盖矩阵（算子 × 场景 → stable / 回退 / 未声明），并对“合法域内回退”报 FAIL；
- 在 CI 上跑，保证“新增算子没声明场景”或“声明了却没接 stable”都能拦住。

### C3. parity 驱动自动生成 + 基线入库

- 由 `scenarios` 生成用例（替代手写），执行后把每个场景的 `diff` 写入
  `tests/stable_scenarios.json` 作为回归基线；
- 任何一次改动导致的场景丢失/数值变化都会在 diff 里显形。

### C4. 运行时回退可视化

`FLA_NPU_THIN_TRACE=1` 时，任何一次回退都打印/计数“算子 + 场景标签”；
CI 在合法域内断言计数为 0。

## 6. Phase D：切换与发布

1. 默认 stable（已改）；`FLA_NPU_THIN_ABI=pybind` 仅保留作对照（需手工构建扩展）。
2. 删除 `_C_thin`：从 wheel 构建与 `setup.py` 默认路径移除；`_thin.py` 保留但不会
   被选中（`_get_thin_op` 已探测扩展，缺失即返回 None）。
3. 发布矩阵：
   - Python 版本无关（`py3-none-any`，实测）；
   - torch 版本：下限约束而非精确 pin（实测可低到 2.7.1），元数据写 `torch>=2.7.1`，
     运行期用 `aoti_torch_abi_version()` + 符号存在性做一次检查；
   - SOC/架构仍分（OPP 的约束，与 torch/Python 无关）。
4. 回归清单（每条都要有命令与预期结果）：覆盖矩阵 0 缺口；parity 基线无 diff；
   跨版本加载矩阵（2.7.1 / 2.9 及后续新版本）；多线程多 stream 事件归属；
   安装态 smoke + 客户场景脚本（含 varlen 与 flag 组合）。

## 7. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| dispatcher 逐参数成本（~2 µs/张量）是硬下限 | 11 张量参数算子 ~1.15–1.27× pybind | 用“stable 直连 vs vllm 量级线”作为主验收；若必须压低，考虑合并算子/减少可选参数 |
| 轻 wrapper 算子（GDR 类）比值偏高 | 公共路径 +0.6–1.2 ms/step（30 次调用） | Phase B 的 B2/B4；并把该比值写进文档而非隐藏 |
| 上游内核坏域（solve_tri ntd 等） | 无法“覆盖” | 记录为内核问题，stable 与 ctypes 行为一致 |
| 校验语义 | 若强求同型报错 → 热路径 +0.03–0.05 ms | 采用 vllm 的分工：schema + 算子校验；全量校验仅在 `FLA_NPU_THIN_VALIDATE=1` |
| NPU 内部 format 张量 | stable 无法表达 | 现网统一 `allow_internal_format=False`（ND）；如需支持，Python 侧查 `get_npu_format` 以 code 传入 descriptor |

## 8. 工作量与推进顺序

| 阶段 | 内容 | 估时 |
| --- | --- | --- |
| A1 | 门面接线（8+5 个算子） | 1–2 天 |
| A2 | 两个复合 spec | 1–2 天 |
| A3 | conv1d 家族（含 #390 ABI） | 1 天（等 #390） |
| C1–C2 | scenarios + 覆盖门禁 | 1–2 天 |
| B2/B4 | int 缓存 + 校验分层 | 1 天 |
| B5/C3 | 全量 A/B + parity 基线入库 | 2 天（机器时间为主） |
| D | 切换与发布清理 | 1 天 |

建议顺序：A1 → C1/C2（先把门禁立起来，后续每个算子接入即被记录）→ A2/A3 → B2/B4
→ B5/C3 → D。
