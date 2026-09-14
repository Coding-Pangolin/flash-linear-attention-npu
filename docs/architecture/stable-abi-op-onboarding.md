# 新增算子适配（Stable-ABI 薄层）

一次适配 = **2 处 C++ + 1 处 Python + 2 行注册**，外加验证件。设计背景见 [stable-abi-macro-design.md](stable-abi-macro-design.md)。

## 1. 交付件清单

| # | 位置 | 内容 | 必改 |
| --- | --- | --- | --- |
| 1 | `csrc_stable/src/stable_<family>.cpp` | `kSchema_<op>` + `run_<op>`（申请输出 + 一条 `FLA_STABLE_EXEC`） | ✅ |
| 2 | `csrc_stable/src/stable_ops.cpp` | `m.def(kSchema_<op>);` + `m.impl("<op>", &boxed_adapter<run_<op>>);` | ✅ |
| 3 | `fla_npu/ops/ascendc/_stable.py` | 一个真签名 wrapper（`_op("<op>")(...)`） | ✅ |
| 4 | `fla_npu/ops/ascendc/__init__.py` | 仅当算子原地写参数：`MUTATED_ARGUMENTS` 加一行（必要时 `MUTATION_FLAGS`） | 视情况 |
| 5 | `tests/regression_thin_ops.py` | 一个 parity 场景（ctypes vs launcher 逐位）+ 在场景列表登记 | ✅ |
| 6 | `tests/stable_scenarios.json` | 场景基线（`FLA_NPU_BASELINE_WRITE=1` 跑一次写入） | ✅ |
| 7 | `tools/stable_ctypes_fallbacks.py` | 若曾登记过该算子的回退：删除条目 | 视情况 |

公开 API 名字不需要加到任何白名单：`__init__.py` 的 `_get_stable_op(name)` 就是 `getattr(_stable, name, None)`，有同名函数即走薄层，没有才回落 ctypes 并在 `BACKENDS` 里记一笔。

## 2. 模板

### 2.1 简单形态：`npu_kda_gate_cumsum`（24 行）

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

### 2.2 条件输出：`npu_chunk_fwd_h`

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

### 2.3 字符串参数：名表 + code

```cpp
constexpr const char* kChunkKdaFwdLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};
...
cstr(kChunkKdaFwdLayoutNames, layout)     // int code → const char*
```

```python
        _char_code("npu_chunk_kda_fwd", "layout", layout)   # str → int code（_stable._ENUM）
```

顺序必须与 `_stable._ENUM` 一致；layout 统一用 `BSND, BNSD, TND, NTD`，并用 `thin_stable/layout_math.h` 算 token/head/dim/chunk。

### 2.4 需要本地策略的算子

`npu_chunk_kda_bwd` 这类带设备相关 workaround 的算子，C++ 适配只做**一次 launch**，多调用/切分/补齐的逻辑留在 Python wrapper（它决定发几次调用），例如：

```python
    if (is_a2_device and use_dense_varlen_fallback) or (is_a5_device and has_varlen_tail):
        ... 逐序列 dense 调用后按 token 轴拼接、标量梯度求和
```

## 3. 落地步骤

```bash
# 1. 写适配（上面的 1-3），登记（4）
# 2. 离线门禁
python torch_custom/fla_npu/tools/stable_coverage.py          # 覆盖 + 枚举表
python torch_custom/fla_npu/tools/op_abi_parity.py            # schema vs 适配函数
python torch_custom/fla_npu/tools/op_api_parity.py            # 公开签名 vs ctypes
python torch_custom/fla_npu/tools/stable_ctypes_fallbacks.py  # 不许回退
python -m unittest tests.test_stable_gates                    # 门禁自测
# 3. 与 OPP 头文件对拍（需要装了 OPP 的机器）
python torch_custom/fla_npu/tools/op_abi_validate.py \
    --opp-include <opp>/op_api/include/aclnnop <cann>/include/aclnnop
# 4. 编 .so 并跑 parity
python csrc_stable/build_stable.py --out /path/libfla_npu_thin.so --no-debug-probe
FLA_NPU_STABLE_LIB=/path/libfla_npu_thin.so PYTHONPATH=<env> \
    python tests/regression_stable_full.py
```

## 4. 新增场景的最低矩阵（T2）

按算子形态取轴，不要求一次全给，但基线里的场景集合**只能增不能减**：

- **布局**：该算子声明的每个 layout（`_ENUM` 里的全部取值）；
- **序型**：dense / varlen（`cu_seqlens`，必要时 canonical `chunk_indices`）/ 物理 B=1；
- **可选参数**：全给、全不给、逐个单给；
- **flag**：每个布尔参数各翻转一次（含决定条件输出的那个）；
- **dtype**：算子支持的每种；
- **非连续**：state / conv_state 带 stride 的情况；
- **边界**：T=1、chunk_size 最小、batch=1、单 chunk、空 tensor 与 `None`；
- **错误路径**：device/dtype/shape/枚举 code/int[] dtype 非法时两侧都拒绝（错误类型允许不同型）。

## 5. 常见坑

- `int[]` 只能是 **host** int32/int64 tensor：写 `_host_ints(...)`，device tensor 会被 C++ 侧拒绝。
- 原地写参数的算子必须登记 `MUTATED_ARGUMENTS`，否则 autograd 看到的是被改过的输入却没有版本号。
- workspace 的设备从**第一个 NPU 输入的 meta** 取；拿错设备会在别的卡上分配。
- stream 每次调用现取（`_current_stream_ptr()`），**不要缓存**：vLLM 是多线程多 stream，缓存过的 pointer 会把 kernel 发到别的线程的 stream 上。
- 输出 shape 规则要照抄 ctypes 参考实现（`_aclnn_ctypes.py` 同名函数），包括 dtype（例如 `o` 跟 `v`、state 跟 `q`）和可选输出的存在条件。
