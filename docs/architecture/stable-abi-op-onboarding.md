# 新增算子适配（Stable-ABI 薄层）

一次适配 = **改 1 个族文件 + 2 行注册 + 1 个 Python wrapper**，外加验证件。**不要求写 ctypes 适配**：ctypes 只是回退后端，新算子默认没有它，按 §8 声明即可。设计背景见 [stable-abi-macro-design.md](stable-abi-macro-design.md)。

## 1. 交付件清单

| # | 位置 | 内容 | 必改 |
| --- | --- | --- | --- |
| 1 | `csrc/src/stable_<family>.cpp`（**已有族就加进那个文件**，族表见 §2） | `kSchema_<op>` + `run_<op>`（申请输出 + 一条 `FLA_STABLE_EXEC`） | ✅ |
| 2 | `csrc/src/stable_ops.cpp` | `m.def(kSchema_<op>);` + `m.impl("<op>", &boxed_adapter<run_<op>>);` | ✅ |
| 3 | `fla_npu/ops/ascendc/_stable.py` | 一个真签名 wrapper（`_op("<op>")(...)`） | ✅ |
| 4 | `fla_npu/ops/ascendc/__init__.py` | 仅当算子原地写参数：`MUTATED_ARGUMENTS` 加一行（必要时 `MUTATION_FLAGS`） | 视情况 |
| 5 | `fla_npu/ops/ascendc/__init__.py` | 仅当算子**没有** ctypes 回退（新算子的默认情况）：名字加进 `_LAUNCHER_ONLY_OPS`（见 §8） | 视情况 |

公开 API 名字不需要加到任何白名单：`__init__.py` 的 `_get_stable_op(name)` 就是 `getattr(_stable, name, None)`，有同名函数即走薄层，没有才回落 ctypes 并在 `BACKENDS` 里记一笔。

`stable_coverage.py` 的 `fallback` 列写明每个算子出问题时的回退后端：`ctypes`，或 `none`（在 `_LAUNCHER_ONLY_OPS` 里声明过，见 §8）。

## 2. 适配代码放哪个文件

适配代码永远写在 `csrc/src/` 下的**某个族文件**里——不是一算子一文件，也不是一个大文件。所有族文件都被 `stable_ops.cpp` `#include` 进**同一个编译单元**，所以"放哪"只影响阅读与 review，不影响构建；算子名**只**在 `stable_ops.cpp` 里重复（那是注册）。

| 文件 | 拥有的算子族 |
| --- | --- |
| `stable_conv1d.cpp` | conv1d 全族：`npu_causal_conv1d`、`_fn`、`_update`、`_bwd` |
| `stable_gdn.cpp` | gated-delta-rule 的复合/派生：`npu_chunk_gated_delta_rule_fwd`、`_bwd`、`_bwd_finalize`、`_fwd_prepare`、`npu_chunk_fwd_o`、`npu_chunk_gdn_bwd_intra` |
| `stable_fwd_h.cpp` | h/dh 递归族：`npu_chunk_fwd_h`、`npu_chunk_gated_delta_rule_fwd_h`、`npu_chunk_gated_delta_rule_bwd_dhu` |
| `stable_kda.cpp` | KDA 族：`npu_chunk_kda_fwd`、`_bwd`、`_bwd_intra`、`_bwd_recompute`、`npu_kda_gate_cumsum` |
| `stable_chunk.cpp` | 两族共用的 chunk 级工具：`*wy_repr*`、`chunk_scaled_dot_kkt`、`chunk_local_cumsum`、`chunk_bwd_dqkwg`、`chunk_bwd_dv_local`、`recompute_w_u_fwd`、`solve_tri` |
| `stable_fast_gelu.cpp` | `npu_fast_gelu_custom`、`npu_fast_gelu_custom_backward` |
| `stable_recurrent_gdr.cpp`、`stable_recurrent_kda.cpp` | 两个 **pre-macro** 适配，各自一个文件（见下） |
| `stable_ops.cpp` | **只有注册**：`m.def(kSchema_<op>)` + `m.impl("<op>", &boxed_adapter<run_<op>>)`，并负责 `#include` 各族文件 |

判族三十秒能定：看公开名字属于哪一段——`causal_conv1d*` → conv1d；`chunk_gated_delta_rule_*`
的复合/派生 → gdn（其中的 **h/dh 递归三件套** → fwd_h）；`chunk_kda_*` / `kda_*` → kda；其余
chunk 级工具（wy_repr / kkt / cumsum / dqkwg / dv_local / recompute / solve_tri）→ chunk；
`fast_gelu*` → fast_gelu。

**原地参数不影响归族，也不影响用宏**：`Tensor(a!)` 走类型化拆栈是安全的——证据就是上面那两个
recurrent 适配器自己，它们正是用 `to<Tensor>` 拆 `Tensor(a!) state` / `initial_state` 的。
`stable_recurrent_gdr.cpp` / `stable_recurrent_kda.cpp` 仍是 pre-macro 只是历史遗留（当初的理由
是"描述符要作为 bundle 交给 torch_npu 任务队列，宏里没有这个位置"，而宏路径的 `detail::exec`
走的就是同一个 `detail::enqueue_launch`，该理由不成立），会并回宏；新增的 recurrent 型算子
**直接写进对应族文件用宏**，不要照着这两个文件抄。

每个族文件**头部都有一行 `// Owns ...` 注释**，列着它当前拥有的算子：新增算子时先读那行，
放好之后把新名字加进去。归错族的代价只是 review 时要挪一次（不影响构建），但会让下一个人
更难判断，所以请把这一行维护住。

（为什么不合成一个文件：这 9 个族文件合计约 2.1k 行，合并后所有算子的改动都落在同一个文件上，
并行开发会互相冲突；现在按族分开，读一个算子的改动只需要看一个文件。代价是"该放哪"需要
规则——就是上面这张表 + 文件头 `// Owns ...`。）

## 3. 参数类型对照

写适配时按这张表挑 holder（都在 `include/stable/exec.h`，`FLA_STABLE_EXEC` 的行参顺序必须与
schema 形参顺序、以及 aclnn 头文件顺序一致；`op_abi_parity.py` + `op_abi_validate.py` 会查）：

| schema 里怎么写 | C++ 形参 | Python 传什么 | C++ 里用哪个 holder |
| --- | --- | --- | --- |
| `Tensor x` | `Tensor x` | NPU 张量 | `tensor(meta_of(x))`；需要 ND / 逻辑形状时 `nd_tensor(...)`、`logical_tensor(...)` |
| `Tensor? g` | `std::optional<Tensor> g` | `None` 或张量 | `optional_tensor(g)`；配 ND/逻辑形状用 `nd_optional_tensor`、`logical_optional_tensor` |
| `-> Tensor` | 自己分配：`allocate_like` / `allocate_sizes` | — | 传给 aclnn 时 `out_tensor(meta_of(out))`；ND/逻辑形状用 `nd_out_tensor`、`logical_out_tensor` |
| `-> (Tensor, Tensor, Tensor?)` | `std::tuple<Tensor, Tensor, std::optional<Tensor>>` | — | 缺席的输出槽传 `TensorMeta()`（null aclTensor）；`boxed.h` 会打成 boxed optional（**不能传裸 handle**） |
| `int chunk_size`、`float scale`、`bool use_exp2` | `int64_t` / `double` / `bool` | Python int / float / bool（`None` 在 wrapper 里给默认值） | `scalar(...)` |
| `int layout`（枚举） | `int64_t layout` | **字符串**，wrapper 用 `_char_code("<op>", "layout", layout)` 转 code | `cstr(k<Op>LayoutNames, layout)`，名表顺序要与 `_stable._ENUM` 一致 |
| host int 数组（`query_start_loc_cpu` 等） | `std::optional<Tensor>`（host 侧） | `_host_ints(seq)`，或直接给 CPU int64 tensor | `int_array(x)`；要在 C++ 里取用值时 `int_values(x)` |
| `Tensor(a!) state`（原地写） | 照常写 `Tensor` / `std::optional<Tensor>`，宏按 `to<Tensor>` 消费槽位 | 调用方直接传被改写的张量 | 宏路径对原地参数是安全的；登记见 §7 |
| `int stream` | `int64_t stream` | `_current_stream_ptr()`（**每次现取，不缓存**；按下发方式选 accessor，见 §7） | `FLA_STABLE_EXEC` 的第三个实参 |

三条硬限制：**没有字符串类型**（字符串一律"名表 + int code"）；**`int[]` 只收 host 的 int64 CPU
tensor**；**返回的 `Tensor?` 槽必须走 boxed optional**。

## 4. 模板

### 4.1 简单形态：`npu_kda_gate_cumsum`（24 行）

```cpp
constexpr const char* kSchema_kda_gate_cumsum =
    "npu_kda_gate_cumsum(Tensor g, Tensor? A_log, Tensor? dt_bias, "
    "Tensor? cu_seqlens, int chunk_size, bool use_gate_in_kernel, "
    "bool safe_gate, float lower_bound, int stream) -> Tensor";

Tensor run_npu_kda_gate_cumsum(Tensor g, std::optional<Tensor> A_log,
                               std::optional<Tensor> dt_bias,
                               std::optional<Tensor> cu_seqlens,
                               int64_t chunk_size, bool use_gate_in_kernel,
                               bool safe_gate, double lower_bound,
                               int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  Tensor out = allocate_sizes(g_meta.sizes, kFloat, g_meta);   // 输出 shape/dtype 规则
  FLA_STABLE_EXEC("aclnnKdaGateCumsum", g_meta, stream, tensor(g_meta),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), scalar(chunk_size),
                  scalar(use_gate_in_kernel), scalar(safe_gate),
                  scalar(lower_bound), out_tensor(meta_of(out)));
  return out;
}
```

```python
def npu_kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None,
                        cu_seqlens=None, use_gate_in_kernel=False,
                        safe_gate=False, lower_bound=None):
    return _op("npu_kda_gate_cumsum")(
        g, A_log, dt_bias, _host_ints(cu_seqlens), chunk_size,
        False if use_gate_in_kernel is None else bool(use_gate_in_kernel),
        False if safe_gate is None else bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        _current_stream_ptr())
```

要点：`FLA_STABLE_EXEC` 的第一个参数是 aclnn 符号前缀，第二个是 workspace 的设备来源（取第一个 NPU 输入的 meta），第三个是 stream；之后**严格按 aclnn 头文件顺序**。

### 4.2 条件输出：`npu_chunk_fwd_h`

```cpp
constexpr const char* kSchema_chunk_fwd_h =
    "npu_chunk_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g, Tensor? gk, "
    "Tensor? initial_state, bool output_final_state, int chunk_size, "
    "bool save_new_value, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "bool use_exp2, bool state_v_first, int stream) "
    "-> (Tensor, Tensor, Tensor?)";

  std::optional<Tensor> out_final_state;                     // 只有 output_final_state 时分配
  ...
  out_tensor(out_final_state.has_value() ? meta_of(*out_final_state)
                                         : TensorMeta()),  // 缺席 → null aclTensor
```

要点：schema 里的 `Tensor?` 返回槽必须由 `boxed.h::pack` 打成 boxed optional（`from(std::optional<Tensor>)`），写成裸 handle 会段错误。

### 4.3 字符串参数：名表 + code

```cpp
constexpr const char* kChunkKdaFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};
...
cstr(kChunkKdaFwdLayoutNames, layout)     // int code → const char*
```

```python
        _char_code("npu_chunk_kda_fwd", "layout", layout)   # str → int code（_stable._ENUM）
```

顺序必须与 `_stable._ENUM` 一致；layout 统一用 `BSND, BNSD, TND, NTD`，并用 `stable/layout_math.h` 算 token/head/dim/chunk。

### 4.4 需要本地策略的算子

`npu_chunk_kda_bwd` 这类带设备相关 workaround 的算子，C++ 适配只做**一次 launch**，多调用/切分/补齐的逻辑留在 Python wrapper（它决定发几次调用），例如：

```python
    if (is_a2_device and use_dense_varlen_fallback) or (is_a5_device and has_varlen_tail):
        ... 逐序列 dense 调用后按 token 轴拼接、标量梯度求和
```

## 5. 落地步骤

```bash
# 1. 写适配（上面的 1-3），登记（4）
# 2. 离线门禁
python torch_custom/fla_npu/tools/stable_coverage.py          # 覆盖 + 枚举表
python torch_custom/fla_npu/tools/op_abi_parity.py            # schema vs 适配函数
python torch_custom/fla_npu/tools/stable_ctypes_fallbacks.py  # 不许回退
python -m unittest tests.test_stable_gates                    # 门禁自测
# 3. 与 OPP 头文件对拍（需要装了 OPP 的机器）
python torch_custom/fla_npu/tools/op_abi_validate.py \
    --opp-include <opp>/op_api/include/aclnnop <cann>/include/aclnnop
# 4. 编 .so 并跑设备回归
python csrc/build_stable.py --out /path/libfla_npu_stable.so --no-debug-probe
FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
    python tests/stable_abi/test_input_lifetime.py
```

## 6. 设备回归的最低矩阵（T2）

按算子形态取轴，不要求一次全给，但声明过的合法域要真跑一遍：

- **布局**：该算子声明的每个 layout（`_ENUM` 里的全部取值）；
- **序型**：dense / varlen（`cu_seqlens`，必要时 canonical `chunk_indices`）/ 物理 B=1；
- **可选参数**：全给、全不给、逐个单给；
- **flag**：每个布尔参数各翻转一次（含决定条件输出的那个）；
- **dtype**：算子支持的每种；
- **非连续**：state / conv_state 带 stride 的情况；
- **边界**：T=1、chunk_size 最小、batch=1、单 chunk、空 tensor 与 `None`；
- **错误路径**：device/dtype/shape/枚举 code/int[] dtype 非法时明确拒绝。

## 7. 常见坑

- `int[]` 只能是 **host** int32/int64 tensor：写 `_host_ints(...)`，device tensor 会被 C++ 侧拒绝。
- 原地写参数的算子必须登记 `MUTATED_ARGUMENTS`，否则 autograd 看到的是被改过的输入却没有版本号。
- workspace 的设备从**第一个 NPU 输入的 meta** 取；拿错设备会在别的卡上分配。
- stream 每次调用现取（`_current_stream_ptr()`），**不要缓存**：vLLM 是多线程多 stream，缓存过的 pointer 会把 kernel 发到别的线程的 stream 上。
- 取 stream 有**两种** accessor，配错会把 host 开销付成毫秒级（vLLM 场景下 +40~80 ms/step）：
  - 入队（`FLA_STABLE_EXEC` 把 descriptor + workspace 交给 torch_npu 的 task queue，
    `fla_npu_stable_queue_enqueue_available` 为真）→ 用 `_npu_getCurrentRawStreamNoWait`，
    **不排空队列**；顺序由队列自己保证，此时排空纯属白等。
  - 内联直投（队列不可用、或 `FLA_NPU_STABLE_LAUNCH=inline`）→ 用 `_npu_getCurrentRawStream`，
    它会把 host 已经入队的任务排空后再给 stream，这正是内联 kernel 不掉队的原因。
  - 两条 accessor 都不在时兜底 `torch.npu.current_stream().npu_stream`（等价于后者）。
  这三档由 `_stable.py` 的 `_current_stream_ptr()` 内部决定，**算子适配不要自己取 stream**，
  也不要自己判断该用哪一条：`stable_coverage.py` 只认 `_op(..., _current_stream_ptr())`。
- 想现场验证"这次调用究竟落在哪个 stream"，用 `fla_npu.ops.ascendc._stable._last_launch_stream()`
  （读的是 C++ 侧本线程最后一次 launch 的真实 stream），不要去读 Python 层的中间变量。
- 输出 shape 规则以算子自己的文档/内核接口为准（`aclnn_*.h` 与算子 `docs/`），包括 dtype（例如 `o` 跟 `v`、state 跟 `q`）和可选输出的存在条件。

## 8. 没有 ctypes 回退的算子

新算子**不再要求写 ctypes 适配**：ctypes 现在只是回退后端，没有它算子照样交付，这也是新算子的默认形态。

**代码只多一处**：把算子名加进 `fla_npu/ops/ascendc/__init__.py` 的 `_LAUNCHER_ONLY_OPS`。

```python
_LAUNCHER_ONLY_OPS: tuple[str, ...] = (
    "npu_chunk_gated_delta_rule_fwd_v2",
)
```

`stable_coverage.py` 双向卡这条声明：

- 已发布算子既不在 ctypes 里、也不在 `_LAUNCHER_ONLY_OPS` 里 → FAIL（`published but absent from the ctypes fallback`）；
- 在 `_LAUNCHER_ONLY_OPS` 里、但 ctypes 仍然定义它 → FAIL（声明过期）。

**正确性怎么证明**。逐位 parity 对照（ctypes ↔ 薄层）已经删除，参考要自己带，二选一：

- fla 的 PyTorch 实现（首选——算子本来就是为了加速它）；
- 一次性录制的 golden 张量 + 容差（形状固定、数值稳定的算子适用）。

**少了什么**，心里要有数：

| 角色 | 有 ctypes 回退 | 没有 ctypes 回退 |
| --- | --- | --- |
| aclnn 实参顺序 | `op_abi_validate.py` 对着 OPP 头文件逐参对拍（两种算子同一条门禁） | 同左 |
| host 封装正确性 | 出问题可以切 `FLA_NPU_STABLE_ABI=ctypes` 做对照 | 只能靠自带的参考实现 / golden |
| 回退后端 | 薄层不可用时退回 ctypes | 无回退，调用会明确报错 |

另外，wrapper 把参数交给 dispatcher 的姿态本来没人查（dispatcher 按位置解包，同类型参数换序是静默的）。`stable_coverage.py` 现在对每个算子核对一次：**wrapper 传给 `_op(...)` 的位置参数个数等于 schema 声明的形参个数，且最后一个是 stream**。把 launch 拆到辅助函数里的组合算子（如 `npu_chunk_kda_bwd`）跳过这条。
