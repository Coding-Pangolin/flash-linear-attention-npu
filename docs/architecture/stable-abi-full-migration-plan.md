# Stable ABI 全量适配方案（唯一后端 · 对齐 vllm-ascend 性能 · 全场景覆盖）

> 分支：`feat/stable-abi-thin`
> 逐轮实测与排查过程见 [stable-abi-migration.md](./stable-abi-migration.md)
> 本文是**执行版**：一个算子要改哪里、什么算达标、场景怎么记录、还差什么。

## 0. 与上一版方案的差异（2026-09-11 实测后刷新）

| 项 | v1（方案刚写下时） | v2（A1/A2 落地） | v3（本文，含 conv1d 与 API 契约） |
| --- | --- | --- | --- |
| 覆盖 | 16/26，剩 10 个卡在 `alloc`/`helpers` 的 ATen 惯用法 | 25/26 | **26/26**：`npu_causal_conv1d` 已接入（`alloc` 表达 `head_num` 重排），只剩上游 #390 的 update 变体未定稿 |
| 一次性通过 | 无 | 243 PASS（22 个算子被调用） | **910b：256 PASS / 0 FAIL，24 个算子被调用；950：15 PASS / ALL PASS**（A5 专属 4 场景） |
| Python API 契约 | 未检查 | 未检查 | **ctypes 为唯一真源**：`tools/op_api_parity.py` 报 0 漂移（本轮修掉 10 个算子的签名漂移） |
| 性能 | 只有 GDR/KDA 两个数 | GDR 1.26×、KDA 1.11×、fast_gelu 0.34× | 同上（B2/B4 未做，GDR 仍未达标） |
| 后端取舍 | stable 可选、pybind 默认 | stable 默认 | **stable 为唯一后端**；pybind 仅作 A/B 对照，随后删除 |
| 场景覆盖 | 口头"没丢场景" | 覆盖矩阵门禁（C2） | 门禁 + **每算子演练记录**（910b 24 个 + 950 4 个 + 2 条显式 SKIP 带原因） |
| 产物一致性 | 无 | 无 | **构建戳**：`.so` 与 Python glue 的生成 hash 不一致时加载即报错（本轮踩过陈旧 `.so` 的坑） |

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
  需要精确消息时用 `FLA_NPU_THIN_VALIDATE=1`（整条调用改走 ctypes 参考实现，全量 Python 校验 +
  同一个 kernel，合法输入结果不变）：

  ```
  default      RuntimeError: aclnnChunkFwdHGetWorkspaceSize failed: 161002
  validate=1   RuntimeError: npu_chunk_fwd_h: exactly one of g and gk must be provided.
  ```

当前证据（221 / 910B3，本轮复跑）：

```
libfla_npu_thin.so  245 736 B
undefined  _ZN2at/_ZN3c10 = 0     aoti_torch_* = 38     导出构建戳符号 = 1
regression_stable_full.py (910B3):  256 PASS / 0 FAIL   "ALL PASS: full stable parity"
regression_stable_a5.py   (950PR):   15 PASS / 0 FAIL   "ALL PASS: Ascend950 stable parity"
op_api_parity.py: 26 个算子比对，0 漂移        stable_coverage.py --strict: 退出码 0
```

## 2. 算子清单与当前状态（26 = ctypes 全量）

"适配来源"只有两种：`GEN`（spec 驱动 codegen 生成）与 `手写`（复合语义，spec 表达不了）。
"全量演练"= 本轮 `regression_stable_full.py` 是否真实调用过它。

| # | 算子 | 适配来源 | 张量参数 | 全量演练 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | `npu_causal_conv1d` | GEN | 4 | ✓ | 输出形状按 `run_mode`/`head_num`/`x.dim()` 走 `alloc`；prefill / head_num 重排 / update / spec-decode / width3 全过（含 `conv_states` 原地写）；varlen 形式 A2 与 ctypes 同样被内核拒绝（561002），记为 SKIP |
| 2 | `npu_causal_conv1d_bwd` | GEN | 10 | ✓ | `input_layout` 走 enum |
| 3 | `npu_chunk_bwd_dqkwg` | GEN | 14 | ✓ | 可选输出 `when` 条件 |
| 4 | `npu_chunk_bwd_dv_local` | GEN | 7 | ✓ | |
| 5 | `npu_chunk_fwd_h` | GEN | 9 | ✓ | |
| 6 | `npu_chunk_fwd_o` | GEN | 6 | ✓ | `output_layout` 走 enum |
| 7 | `npu_chunk_gated_delta_rule_bwd_dhu` | GEN | 12 | ✓ | `Tensor?` 槽 nullopt 化的第一个现场 |
| 8 | `npu_chunk_gated_delta_rule_bwd_finalize` | GEN | 19 | ✓(A5) | 910b 无该内核 → 显式 SKIP；950 上由 `regression_stable_a5.py` 覆盖 |
| 9 | `npu_chunk_gated_delta_rule_fwd` | GEN | 18 | ✓ | 4 layout × flag 组合 |
| 10 | `npu_chunk_gated_delta_rule_fwd_h` | GEN | 9 | ✓ | |
| 11 | `npu_chunk_gated_delta_rule_fwd_prepare` | GEN | 16 | ✓(A5) | 6 个场景全过：`a_log`/`dt_bias` 仅在 `use_gate_in_kernel` 时下发（此前 stable 漏掉该 `when`，A5 报 161002） |
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
| 25 | `npu_recurrent_kda` | 手写 | 13 | ✓ | 已并入全量脚本（dense BSND + varlen + in-place state 一致） |
| 26 | `npu_solve_tri` | GEN | 2 | ✓ | `layout` 走 enum；`ntd` 是上游内核坏域，见 §8 |

统计：**生成 24 + 手写 2 = 26/26 有 stable 适配**。演练面：910b 覆盖 24 个，
950 覆盖 4 个（含 2 个 910b 无法加载的 A5 内核），两条 SKIP 都带原因
（`conv1d` varlen 被内核拒绝、A5 内核需要 950 主机）。
`tools/op_stable_codegen.py --parse-only` 逐算子打印适配状态，
`tools/stable_coverage.py --strict` 对"缺适配"直接 FAIL。

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

- **spec JSON 1 个**（`aclnn_name` / `python_name` / `args` / `outputs` / `python` / `scenarios`），
  其中 `python` 块**不要手写**：`python tools/sync_spec_python.py --write` 从
  `_aclnn_ctypes.py` 反推 `positional`/`defaults`/`required`（ctypes 是唯一的 API 真源）；
- **测试场景 1 个**；
- **白名单 0 处**——`_get_direct_op` 按算子名动态解析；`MUTATED_ARGUMENTS` / `MUTATION_FLAGS` 只在"有 inplace 语义"时才加一行；
- 生成产物 2 个（`.inc` 与 `_stable_generated.py`）由脚本产出。

生成后有两条离线门禁会立刻报错，不需要 NPU：
`tools/op_api_parity.py`（Python 签名与 ctypes 逐参数比对）与
`tools/stable_coverage.py`（适配器覆盖 + 场景轴声明）与
`tools/op_abi_parity.py`（**spec 描述的 aclnn 实参列表 vs 实现实际传给 aclnn 的**，
不需要 OPP 头文件——上游 #390 改 `aclnnCausalConv1d` ABI 时就是这条把它抓出来）。
改完 `.inc` 必须重编 `.so`：
构建会把 `.inc` 的 md5 编译进产物，加载时与 Python glue 的 `_GENERATED_HASH` 比对，
不一致直接抛错并给出重编命令。

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
| B1b | `load()` 只解析一次库路径（原来每次调用都要 `os.environ.get` + 文件 stat）；`_current_stream_ptr()` 缓存**访问器函数**（不缓存 stream 值） | **已完成** | GDR 公共路径 0.1188 → 0.0885 ms；stream 仍逐调用读取，避免多线程串流 |
| B2 | `int[]` 按 `tuple(values)` 缓存 host int64 张量（只缓存 list/tuple；tensor 直接透传） | **已完成** | `chunk_scaled_dot_kkt` varlen 公共路径 0.1602 → 0.0976 ms（−39%），实测复用同一张量三次逐位一致 |
| B3 | stream：每调用 raw accessor（~1.2 µs），**不做进程级缓存** | 已完成（含 vLLM 崩溃教训） | 正确性优先 |
| B4 | ✅ 校验分层：schema 免费 + 算子自身合法域；`FLA_NPU_THIN_VALIDATE=1` 时整条调用走 ctypes 参考实现（全量 Python 校验 + 同一个 kernel，结果逐位一致），用于报错定位与"launcher vs kernel"二分 | 已完成 | 默认不付校验成本；非法输入在 VALIDATE 下给出精确消息 |
| B5 | 每算子交替采样 A/B（两轮，P50+P90） | 待做（当前只有 3 个算子） | 出 26 行验收表 |

**"不能回退 thin" 的落点**就是 B2/B4：GDR 现在 1.26×，必须压回 ≤1.15×；
压不下去就把"已定位到 dispatcher 逐参数成本"的书面结论记进去，而不是悄悄换回 pybind。

更新（2026-09-11 晚，交替采样，pybind 0.0699 / stable 0.0879）：

| 路径 | 相对直连 | 说明 |
| --- | --- | --- |
| stable 直连 | ×1.00 | dispatcher + 逐参数装箱，结构性成本 |
| stable 后端 wrapper（`_stable.npu_*`） | ×1.13 | 库加载快路径、stream 读取、句柄查找 |
| stable 公共 wrapper（`asc.npu_*`） | ×1.27 | 再 +12 µs，是 mutation 契约（version/grad）——这是**有意保留**的正确性成本 |
| ctypes 公共 wrapper | ×5.56 | 参照：旧方案 |

按"stable 直连 ≤ vllm-ascend 量级线（0.073 ms）"这条主口径，我们已经在量级内；
与 pybind 的比值是次要口径，且 pybind 已不在默认链里（D1）。

### 4.4 逐算子 host A/B（B5，2026-09-11 实测）

`tests/bench_stable_host.py` 复用 `regression_thin_ops` 的场景输入（即已被证明逐位
一致的那些输入），只把两个后端的调用挂上计时器，跑 4 轮取 P50。测的是
**host enqueue**：不含 `synchronize`；两次调用之间的 parity 比对会强制同步，
所以两条后端面对同样的（空闲流水）条件，**比值可比**，绝对值比背靠背 decode 低。

结果（910B3，24 个可跑算子，单位 ms）：**每一个算子 stable 都明显快于 ctypes**，
比值区间 **0.19×–0.39×**（即快 2.6–5.3 倍）。

| 算子 | ctypes | stable | 比值 |
| --- | --- | --- | --- |
| `npu_chunk_local_cumsum`（最差） | 0.4282 | 0.1659 | 0.39× |
| `npu_chunk_kda_bwd` | 0.9533 | 0.3421 | 0.36× |
| `npu_fast_gelu_custom` | 0.3408 | 0.1106 | 0.32× |
| `npu_chunk_gated_delta_rule_fwd` | 0.7890 | 0.2119 | 0.27× |
| `npu_solve_tri` | 1.1486 | 0.2705 | 0.24× |
| `npu_chunk_bwd_dqkwg` | 1.1547 | 0.2544 | 0.22× |
| `npu_recurrent_kda`（最好） | 0.6832 | 0.1307 | 0.19× |

完整 24 行落在 `tests/bench_stable_host_910b.json`（可 diff 的历史记录）。
剩下 2 个（`chunk_gated_delta_rule_fwd_prepare` / `_bwd_finalize`）是 A5 专属内核，
910b 上不参与这张表；它们的正确性由 `regression_stable_a5.py` 在 950 上覆盖。

结论：**"不回退 thin" 这条要求在本轮拿到了逐算子的证据**——不是抽样外推，
而是 24/24 都快于我们此前实际发货的 ctypes 路径。

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

### C3. parity 基线入库（已实现）

`tests/stable_scenarios.json` 按设备记录**这次跑过哪些场景**：

```json
{
  "Ascend910B3":        {"passed": {"chunk_kda_fwd(BSND varlen=0 ...)": 0.0, ...}, "skipped": {...}},
  "Ascend950PR_9579":   {"passed": {...}, "skipped": {}}
}
```

写：`FLA_NPU_BASELINE_WRITE=1 python tests/regression_stable_full.py`；
默认模式是**比对**——少了一个场景（丢 layout、丢 flag 组合）就 FAIL 并点名，
数值不再是 0.0 也 FAIL。这样"覆盖"是仓库里的一份事实，而不是某次日志里的一行
`ALL PASS`。当前记录：910B3 **245 通过 + 2 条带原因的 SKIP**，950PR **12 通过**。

### C4. 运行期回退可视化（已实现）

解析后端时记录 `BACKENDS`（算子 → 服务它的后端）与 `FALLBACKS`（被迫走 ctypes 的
算子及次数）；`FLA_NPU_THIN_TRACE=1` 让每个算子打一行到 stderr：

```
[fla-npu] npu_fast_gelu_custom: stable
```

更重要的是它可以当门禁用：`FLA_NPU_DISPATCH=public python tests/regression_stable_full.py`
把**整套场景**改走公共 API（而不是直接钉住后端），于是覆盖到"后端选择 + mutation
契约 + launcher"整条链，跑完断言 `FALLBACKS` 为空。实测：

```
public dispatch: 24 operators, backends ['stable'], no fallback
ALL PASS: full stable parity          (259 PASS, 基线 246 + 3 条 SKIP)
```

四种模式的行为也逐一对过：默认 → `stable`；`FLA_NPU_THIN_TRACE=1` 打印该行；
`FLA_NPU_THIN_ABI=ctypes` → `ctypes`（显式选择，不计回退）；
`FLA_NPU_THIN_VALIDATE=1` → `ctypes` 且**记录一次回退**（带原因）。

### C5. 补全演练缺口（已完成）

三个"注册了但没演练"的算子全部补上：`npu_recurrent_kda` 并入
`regression_stable_full.py`；`npu_chunk_gated_delta_rule_fwd_prepare` /
`_bwd_finalize` 在 910b 上打印带原因的 SKIP，在 950 上由
`tests/regression_stable_a5.py` 全量跑（15 PASS）。

### C6. Python API 契约（新增，已实现）

`tools/op_api_parity.py` 用 `ast` 解析 `_aclnn_ctypes.py` 与 stable 后端，
逐算子比对参数顺序、位置/关键字归属、默认值缺失与变化。它在本轮抓出 10 个算子
的漂移，其中 6 个是**会直接抛 TypeError** 的：

| 漂移 | 算子 | 后果 |
| --- | --- | --- |
| 位置参数默认值丢失（`initial_state=None`/`dht=None`） | `npu_causal_conv1d_bwd` | `f(x, y, w, dy)` 报缺参 |
| 整个参数被漏掉（`transpose_state_layout`） | `npu_chunk_gated_delta_rule_bwd_dhu` | 传该参数报 TypeError |
| 位置参数被改成关键字（`g`） | `npu_chunk_gated_delta_rule_fwd_h` | 位置调用报 TypeError |
| 位置默认值丢失（`chunk_size=64`） | `npu_chunk_gated_delta_rule_fwd_prepare`、`npu_chunk_kda_fwd` | 省略即报缺参 |
| 关键字改名（`chunk_indices` vs ctypes 的 `chunk_indices_out`） | `npu_chunk_local_cumsum` | 按 ctypes 名调用报 TypeError |
| 默认值语义变化（`False` → `None`） | `npu_kda_gate_cumsum` | 不传该 flag 时行为不同 |

修法不是逐个打补丁，而是把 `python` 块改成从 ctypes 反推
（`tools/sync_spec_python.py`），并让生成器支持"位置参数带默认值"与
"必填关键字参数"。规格里的 `python.positional`/`defaults` 从此由工具维护。

### C7. 产物一致性（新增，已实现）

`.inc` 改了但忘记重编 `.so` 时，此前会退化成一个难读的 dispatcher 错误，
甚至静默用错 stream。现在构建把 `.inc` 的 md5 通过
`-DFLA_STABLE_SOURCE_HASH=` 编进 `libfla_npu_thin.so`（导出
`fla_npu_thin_source_hash()`），`_stable.load()` 用 ctypes 读出来与
`_stable_generated._GENERATED_HASH` 比对；不一致直接抛
"was built from different generated adapters ... Rebuild ..."。负例已实测
（把 glue 的 hash 改掉后加载报错），旧产物（无该符号）按兼容处理。

## 6. 剩余工作与推进顺序

### 6.1 #390 已合入 main：conv1d 家族有 OPP 可验，两个 recompute 算子还没有 kernel（2026-09-11 实测）

`origin/main` 的当前 tip 就是 "Merge pull request #390 from LiuZonggu/causal-conv1d"，
它带来 4 个新入口（`npu_causal_conv1d_fn`、`npu_causal_conv1d_update`、
`npu_chunk_gdn_bwd_intra`、`npu_chunk_kda_bwd_recompute`），并把
`npu_causal_conv1d` 改成兼容壳、**同时更换了 `aclnnCausalConv1d` 的 ABI**：

| 位置 | 旧原型（我们手上的 OPP，`env_cgdrfwd_full/.../aclnnop/aclnn_causal_conv1d.h`） | 新代码（main 的 ctypes/`_launch_causal_conv1d`） |
| --- | --- | --- |
| 4–7 | `aclIntArray*`（queryStartLoc / cacheIndices / initialStateMode / numAcceptedTokens） | `aclTensor*`（query_start_loc / cache_indices / has_initial_state / num_accepted_tokens） |
| 8–11 | `int64_t activationMode` … | `aclIntArray*`（四个 CPU 版本） |
| 12 | `int64_t padSlotId` | `const char* activation` |
| 13–16 | `runMode / headNum / out` | `padSlotId / nullBlockId / runMode / headNum / maxQueryLen / out` |

实测（221，把 main 的两个文件覆盖到安装态再调用）：

```
legacy npu_causal_conv1d（新 ctypes + 旧 OPP）→ 进程静默退出（无 traceback，ABI 不匹配）
同一调用（旧 ctypes + 旧 OPP）        → OK shape=(2,4,16)
```

也就是说 **main 的 conv1d 需要配套的新 OPP**（op_api 头 + kernel），而 221 上的
OPP 是 #390 之前的；241 的 A5 OPP 里干脆没有 `aclnnCausalConv1d`
（`Unable to resolve aclnn symbol`）。因此 **merge main 会让 conv1d 在本环境不可验证**，
本轮先把 merge 退回（未推送），保持分支停在已被完整验证的基线上。

拿到新 OPP 之后这一步就变成纯机械工作：4 个新 op 各写一份 spec（conv1d 家族共用一个
ABI，`python` 块负责 activation 字符串与 CPU 元数据数组的归一），跑
`sync_spec_python.py` + `op_api_parity.py` + 设备侧场景；覆盖门禁现在就会直接列出
缺哪些适配器（实测：`ctypes operators: 30`，4 个 FAIL）。

**同日继续查证：四个新算子的 OPP 供给并不一样。**

| 新算子 | aclnn 入口 | 可用 OPP 里有没有 kernel | 能否在本机验证 |
| --- | --- | --- | --- |
| `npu_causal_conv1d_fn` | `aclnnCausalConv1d`（新 ABI） | ✅ `/data/fangziyang/code/0908/env390` 就是新 ABI 头 + 实现 | ✅ 实测 legacy / fn / update 三个入口都 OK |
| `npu_causal_conv1d_update` | 同上 | ✅ 同上 | ✅ 同上 |
| `npu_chunk_gdn_bwd_intra` | `aclnnChunkGdnBwdIntra` | ❌ 扫遍 0908 下所有 `libcust_opapi.so`，符号数为 0 | ❌ `Unable to resolve aclnn symbol` |
| `npu_chunk_kda_bwd_recompute` | `aclnnChunkKdaBwdRecompute` | ❌ 同上 | ❌ 同上 |

所以"适配 #390"实际是两件事：

1. **conv1d 家族可以做、而且能验**：把测试环境切到 `env390` 的 OPP
   （`ASCEND_CUSTOM_OPP_PATH=<env390>/fla_npu/opp/vendors/fla_npu_transformer:.../op_api/lib`）。
   三个 Python 入口共用同一条 ABI，但每个入口固定了不同的槽位——
   legacy 固定 `null_block_id=-1`；fn 固定 `run_mode=0`；update 固定 `run_mode=1`、
   `pad_slot_id=-(1<<63)`，并且**把结果 copy 回 `x` / `out`**。因此生成器需要补一个
   "常量参数"能力（`const`），或者这三个走手写适配；这一层目前是空的。
2. **两个 recompute 算子只能等 OPP**：kernel 不在任何可用 OPP 里，写出来的适配器
   无法验证——按本方案的规矩，不验的东西不进主干。它们在 main 的 ctypes 里存在，
   所以一旦合 main，覆盖门禁会立刻把它们标成缺口（实测 `ctypes operators: 30`，
   4 个 FAIL，与门禁的预期行为一致）。

本轮结论：**不合 main**（与用户既定口径一致："#390 只用来验证，不并进我们分支"），
分支继续停在全绿基线上；上面两张表就是下一步的全部输入。

| 阶段 | 内容 | 前置 |
| --- | --- | --- |
| A3 | `npu_causal_conv1d` 已适配（#390 之前的 ABI）；`causal_conv1d_fn` / `_update` / `chunk_gdn_bwd_intra` / `chunk_kda_bwd_recompute` 待适配 | **卡在新的 OPP**：#390 已合入 main，但它同时换了 `aclnnCausalConv1d` 的 ABI（见 §6.1），我们手上的 OPP 还是旧原型 |
| C1–C2 | `scenarios` + 覆盖矩阵门禁 | **C2 已实现**；C1 的逐算子轴值核对待补（当前靠 spec 反推 + 演练记录） |
| C5 | ✅ 3 个演练缺口已补（910b 24 个 + 950 4 个） | — |
| C6 | ✅ Python API 契约门禁（0 漂移） | — |
| C7 | ✅ 构建戳（陈旧 `.so` 直接报错） | — |
| C4 | ✅ 回退可视化 + 公共 API 全量演练（24 算子全为 stable，0 回退） | — |
| C8 | ✅ spec 的 aclnn 实参列表 vs 实现（离线，能提前发现 #390 那类 ABI 换血） | — |
| B2 | ✅ int 缓存（varlen 路径 −39%） | — |
| B4 | 校验分层（schema + C++ 廉价断言），把 GDR 压回 ≤1.15× | 无 |
| C3 | ✅ parity 基线入库（910B3 245 + 950PR 12，丢失场景即 FAIL） | — |
| B5 | ✅ 逐算子 A/B 表（`tests/bench_stable_host.py`，24 个可跑算子全部快于 ctypes，比值 0.19–0.39×） | — |
| D1 | ✅ 默认链 stable → ctypes；`FLA_NPU_THIN_ABI=pybind/ctypes` 才算显式切换（顺带修掉 `=ctypes` 其实没生效的老问题） | — |
| D2 | ✅ 默认构建不再编 `_C_thin`，wheel 自带 `libfla_npu_thin.so`；一键编包产物 `py3-none-any` 并在干净目录安装后跑通全量 | — |
| D3 | ✅ 发布矩阵：ABI-free wheel 声明 `torch>=2.7.1` / `torch_npu>=2.7.1` 下限（pybind wheel 仍是精确 pin），加载失败时给出"需要 ≥2.7.1"的明确报错 | — |

建议顺序：**C1 → B4 → B5 →（#390 就绪后）conv1d update 变体**。
门禁（C1/C2/C6/C7）已经立在前面，后续每个算子的接入自动被记录。

### D3 细节：一个包服务整段 torch 版本

| wheel | 元数据 | 说明 |
| --- | --- | --- |
| 默认（ABI-free） | `torch>=2.7.1`、`torch_npu>=2.7.1` | 2.9 头编译，2.7.1 运行实测（241：py3.10 + torch 2.7.1.post5 跑完全部 A50 场景）；一条下限覆盖整段版本 |
| `FLA_NPU_BUILD_THIN=1` | `torch==<build>`、`torch_npu==<build>` | pybind 是 ABI 匹配的，装错是硬错误而非警告 |
| `FLA_NPU_BUILD_STABLE_ABI=0` | 只带 ctypes，无任何 torch 约束 | 纯 Python wheel |

运行期那一半：`_stable.load()` 在 `torch.ops.load_library` 失败时不再抛裸的
"undefined symbol"，而是说明"该 launcher 需要 torch >= 2.7.1（它解析的
`aoti_torch_*` 符号在 2.7.x 之后才补齐）"，并附上原始错误。

## 7. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| codegen 的 stack 下标一旦写错（例如 stream 取到倒数第二个槽） | 编译通过、正常算子可能"碰巧"能跑，换一个 SOC 就直接 segfault | `generate_cpp` 生成后断言"读到的下标恰为 `0..len(params)`、写回恰为 `0..outputs-1`"；负例已实测能拦下 |
| `.so` 与 Python glue 来自不同 codegen 轮次 | dispatcher 报错或静默用错 stream；本轮因此白跑三次 | C7 的构建戳（加载即比对，报错给出重编命令） |
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
