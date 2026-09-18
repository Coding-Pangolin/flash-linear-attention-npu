# fla_npu 适配层

`torch_custom/fla_npu` 把 Ascend C 算子变成可调用的 Python 接口。上层只认一个入口：

```python
from fla_npu.ops import ascendc as ascendc_ops

out = ascendc_ops.npu_chunk_fwd_o(...)

# 等价的短名（去掉 npu_ 前缀）
from fla_npu.ops.ascendc import chunk_fwd_o

out = chunk_fwd_o(...)
```

## 1. 现状

**默认后端是 `csrc/` 编出的 Stable-ABI 薄层**（`libfla_npu_stable.so` + `_stable.py`），当前 31 个公开算子全部由它承载，张量走 torch dispatcher。**ctypes**（`_aclnn_ctypes.py`）退居**参考实现与回退**：怀疑薄层有问题时用 `FLA_NPU_STABLE_ABI=ctypes` 走它做对照，离线门禁也拿它当参考给薄层做逐位 parity。

薄层不绑 CPython ABI、也不绑 libtorch C++ ABI，一份 wheel 可跨 Python / torch 版本使用（最低已验证 torch 2.7.1）；host 开销与 vLLM-Ascend 的 custom 路径同一量级。

| 路径 | 职责 |
| --- | --- |
| `csrc/include/stable/exec.h` | `FLA_STABLE_EXEC`：参数 holder（保活到 launch 之后）、符号解析、workspace、下发 |
| `csrc/include/stable/boxed.h` | `boxed_adapter<run_*>`：按 `run_*` 的签名拆 boxed 栈、按返回值类型打包 |
| `csrc/include/stable/{acl_meta,runtime,layout_math,at_facade}.h` | 张量元信息与输出分配、workspace / stream、layout（BSND / BNSD / TND / NTD）换算、可用的 torch C API 门面 |
| `csrc/src/stable_<family>.cpp` | 各算子族的适配：一个算子 = 一条宏（族表见 §2.5） |
| `csrc/src/stable_ops.cpp` | 只做注册（`m.def` + `m.impl`），并把各族文件 include 进同一个编译单元 |
| `fla_npu/ops/ascendc/_stable.py` | 薄层的 Python wrapper（真签名）、后端选择、取 stream |
| `fla_npu/ops/ascendc/_aclnn_ctypes.py` | ctypes 参考实现 |
| `fla_npu/ops/ascendc/__init__.py` | 公开入口、短名导出、正反向绑定、mutation 契约 |
| `tools/*.py` | 离线门禁 |
| `tests/stable_abi/` | 需要 NPU 的设备回归 |

运行期开关：

| 环境变量 | 取值 | 作用 |
| --- | --- | --- |
| `FLA_NPU_STABLE_ABI` | `ctypes` | 强制走 ctypes 参考路径（对照 / 诊断） |
| `FLA_NPU_STABLE_VALIDATE` | `1` | 每次调用改走 ctypes 参考：非法输入给出精确的 Python 报错。诊断开关，不是性能模式 |
| `FLA_NPU_STABLE_LIB` | 路径 | 指定要加载的 `libfla_npu_stable.so`；不设时用包里那份 |
| `FLA_NPU_STABLE_TRACE` | `1` | 逐算子打印实际后端与回落原因 |
| `FLA_NPU_STABLE_STREAM` | `accessor` | 逃生阀：强制用会排空任务队列的取流方式（见 §3） |
| `FLA_NPU_STABLE_LAUNCH` | `inline` | 逃生阀：不走 torch_npu 任务队列，内联直投 |
| `FLA_NPU_BUILD_STABLE_ABI` | `0` | 构建开关：产出不含薄层的纯 ctypes wheel |

## 2. 新增算子适配

一次适配 = **1 个族文件 + 2 行注册 + 1 个 Python wrapper**，再加一个 parity 场景。**不需要先写一份 ctypes 适配**。更细的规则（条件输出、字符串名表、需要 Python 侧策略的组合算子）见 [`docs/architecture/stable-abi-op-onboarding.md`](../../docs/architecture/stable-abi-op-onboarding.md)，设计取舍见 [`stable-abi-macro-design.md`](../../docs/architecture/stable-abi-macro-design.md)。

### 2.1 用宏，别写手写入口

薄层里每个算子的 C++ 适配只有三件事：取张量、申请输出、下发一次 aclnn。这三件事与算子无关，随算子变的只有"第几个参数是什么类型"，所以宏把前三件做掉，只留参数表：

- `FLA_STABLE_EXEC(aclnn 符号前缀, workspace 设备来源, stream, 按 aclnn 头文件顺序的实参...)`：解析符号、由实参推导 `GetWorkspaceSize`、建 workspace、下发，并把失败转成 C++ 异常。
- `boxed_adapter<run_*>`：按 `run_*` 的签名拆栈、按返回值打包。有了它，`stable_ops.cpp` 里注册一个算子只要两行。

手写入口要自己做这两件事，也就绕过了 boxed kernel 的输入所有权契约（`torch/csrc/stable/library.h:69`：*fn is responsible for stealing the memory of the inputs, in effect "popping" them off the stack*）。历史教训就在这里：`npu_recurrent_gated_delta_rule` / `npu_recurrent_kda` 最初用手写入口、用 `to<AtenTensorHandle>` 读必选槽——那是纯重解释、不消费栈引用，于是每次调用漏一份引用：2000 次 decode 形状调用让 caching allocator 涨 191 MiB，服务级涨到 8.2 GiB 后 OOM。宏用 `to<Tensor>` 消费每个必选 Tensor 槽，这类错误在类型层面就写不出来。

### 2.2 交付件

| # | 位置 | 内容 |
| --- | --- | --- |
| 1 | `csrc/src/stable_<family>.cpp`（已有族就加进那个文件，族表见 §2.5） | `kSchema_<op>` + `run_<op>`：申请输出 + 一条 `FLA_STABLE_EXEC` |
| 2 | `csrc/src/stable_ops.cpp` | `m.def(kSchema_<op>);` + `m.impl("<op>", &boxed_adapter<run_<op>>);` |
| 3 | `fla_npu/ops/ascendc/_stable.py` | 一个真签名 wrapper：`_op("<op>")(...)` |
| 4 | `fla_npu/ops/ascendc/__init__.py` | 仅当算子原地写参数：`MUTATED_ARGUMENTS` 加一行（必要时 `MUTATION_FLAGS`） |
| 5 | `tests/stable_abi/regression_ops.py` | 一个 ctypes ↔ 薄层的逐位 parity 场景，并登记进场景列表 |
| 6 | `tests/stable_abi/stable_scenarios.json` | 场景基线，`FLA_NPU_BASELINE_WRITE=1` 跑一次写入 |

公开名不用加白名单：`_get_stable_op(name)` 就是 `getattr(_stable, name, None)`。算子**没有** ctypes 参考时，把名字加进 `__init__.py` 的 `_LAUNCHER_ONLY_OPS`，参考换成 fla 的 torch 实现或一次性录制的 golden 张量（规则见接入文档 §8）。

### 2.3 骨架

C++（`csrc/src/stable_kda.cpp`）：

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
  Tensor out = allocate_sizes(g_meta.sizes, kFloat, g_meta);
  FLA_STABLE_EXEC("aclnnKdaGateCumsum", g_meta, stream, tensor(g_meta),
                  optional_tensor(A_log), optional_tensor(dt_bias),
                  int_array(cu_seqlens), scalar(chunk_size),
                  scalar(use_gate_in_kernel), scalar(safe_gate),
                  scalar(lower_bound), out_tensor(meta_of(out)));
  return out;
}
```

Python（`fla_npu/ops/ascendc/_stable.py`）：

```python
def npu_kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None,
                        cu_seqlens=None, use_gate_in_kernel=False,
                        safe_gate=False, lower_bound=None):
    return _op("npu_kda_gate_cumsum")(
        g, A_log, dt_bias, _host_ints(cu_seqlens), chunk_size,
        bool(use_gate_in_kernel), bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        _current_stream_ptr())
```

四条顺序必须一致，门禁会查：`kSchema_<op>` 形参 === `run_<op>` 形参 === `FLA_STABLE_EXEC` 实参 === aclnn 头文件顺序（`stream` 固定在最后）。schema 的形参顺序同时就是 Python wrapper 传给 `_op(...)` 的位置参数顺序，`stable_coverage.py` 会核对个数。

### 2.4 参数怎么写

| schema | C++ 形参 | C++ holder | Python 侧 |
| --- | --- | --- | --- |
| `Tensor x` / `Tensor? g` | `Tensor` / `std::optional<Tensor>` | `tensor(meta_of(x))` / `optional_tensor(g)`；要 ND 或逻辑形状时用 `nd_*` / `logical_*` | 张量；可选参数传 `None` |
| `->` 单个 `Tensor` | 自己分配 | 交给 aclnn 时用 `out_tensor(meta_of(out))` | — |
| `->` 多个（含 `Tensor?`） | `std::tuple<...>` | 缺席的输出槽传 `TensorMeta()`（null aclTensor），由 `boxed.h` 打成 boxed optional | — |
| `int` / `float` / `bool` | `int64_t` / `double` / `bool` | `scalar(...)` | int / float / bool；`None` 在 wrapper 里给默认值 |
| `int layout`（枚举） | `int64_t layout` | `cstr(k<Op>LayoutNames, layout)` | 字符串 → `_char_code(...)` 转 int code，两边名表顺序必须一致 |
| host int 数组 | `std::optional<Tensor>`（host） | `int_array(x)`，取值用 `int_values(x)` | `_host_ints(seq)`，或直接给 CPU int64 tensor |
| `int stream` | `int64_t stream` | `FLA_STABLE_EXEC` 的第三个实参 | `_current_stream_ptr()`；每次现取，不缓存 |

三条硬限制（StableIValue 没有对应类型）：**没有字符串**，一律"名表 + int code"；**`int[]` 只能用 host int64 CPU tensor**，device tensor 会被 C++ 侧拒绝；**返回的 `Tensor?` 必须走 boxed optional**，写裸 handle 会段错误。

原地写参数的算子（`conv1d` 的 `conv_state`、recurrent 的 `state`）要登记 `MUTATED_ARGUMENTS`，否则 autograd 看不到版本变化；`inplace_final_state=False` 这种"有时不改"的形态用 `MUTATION_FLAGS`。

### 2.5 放哪个族文件

适配写在 `csrc/src/` 下的**族文件**里，不是一算子一文件，也不是一个大文件。所有族文件都被 `stable_ops.cpp` include 进同一个编译单元，"放哪"只影响阅读与 review：

| 文件 | 算子族 |
| --- | --- |
| `stable_conv1d.cpp` | `npu_causal_conv1d` / `_fn` / `_update` / `_bwd` |
| `stable_gdn.cpp` | GDN 的复合与派生算子 |
| `stable_fwd_h.cpp` | h / dh 递归族 |
| `stable_kda.cpp` | KDA 族 |
| `stable_chunk.cpp` | 两族共用的 chunk 级工具（wy_repr / kkt / cumsum / dqkwg / dv_local / recompute_w_u_fwd / solve_tri） |
| `stable_fast_gelu.cpp` | `npu_fast_gelu_custom` 正反向 |
| `stable_recurrent_gdr.cpp` / `stable_recurrent_kda.cpp` | 两个 recurrent 适配，各自一个文件（见下） |
| `stable_ops.cpp` | **只有注册** |

每个族文件头部有一行 `// Owns ...` 注释列出它拥有的算子，新增前先读它，放好之后把新名字加进去。

> 遗留：`stable_recurrent_gdr.cpp` / `stable_recurrent_kda.cpp` 是仅存的两个手写入口，会并回宏。**新算子一律用宏**。

### 2.6 门禁与场景

```bash
# 离线门禁
python torch_custom/fla_npu/tools/stable_coverage.py          # 覆盖、枚举表、wrapper 位置参数个数
python torch_custom/fla_npu/tools/op_abi_parity.py            # schema vs 适配函数 vs ctypes
python torch_custom/fla_npu/tools/op_api_parity.py            # 公开签名 vs ctypes
python torch_custom/fla_npu/tools/stable_ctypes_fallbacks.py  # 不许回退
python -m unittest tests.test_stable_gates

# 与 OPP 头文件逐参对拍（需要装了 OPP 的机器）
python torch_custom/fla_npu/tools/op_abi_validate.py \
    --opp-include <opp>/op_api/include/aclnnop <cann>/include/aclnnop

# 编 launcher 并跑设备 parity
python torch_custom/fla_npu/csrc/build_stable.py --out /path/libfla_npu_stable.so --no-debug-probe
FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
    python tests/stable_abi/regression_stable_full.py
```

场景矩阵按算子形态取轴：该算子声明的每个 layout × dense / varlen / 物理 B=1 × 可选参数（全给、全不给、逐个单给）× 每个 bool 翻转（含决定条件输出的那个）× dtype × 非连续 state × 边界（T=1、batch=1、单 chunk、空 tensor / `None`）× 错误路径（device / dtype / shape / 枚举 code / `int[]` dtype 非法时两侧都拒绝）。基线里的场景**只能增不能减**。

## 3. 注意事项

- **stream 每次现取，绝不要缓存。** 进程级缓存过一个 stream pointer，在 vLLM 的多线程多 stream 下会把 kernel 发到别的线程的 stream 上（512 那次崩溃就是这么来的）。适配代码不要自己取 stream，也不要把值存下来。
- **取 stream 的 accessor 要和下发方式配对。** 下发走 torch_npu 任务队列（vLLM `EXEC_NPU_CMD` 同路）时用不排空队列的 accessor，顺序由队列保证；内联直投时必须用会排空队列的那把，否则 kernel 会插到已入队任务的前面。配错在空队列上看不出来，在 vLLM worker 上会变成每次调用约 1 ms。这段逻辑固定在 `_stable.py::_current_stream_ptr()` 里，两个逃生阀是 `FLA_NPU_STABLE_STREAM=accessor` 与 `FLA_NPU_STABLE_LAUNCH=inline`。
- **launcher 的加载时机。** 由 `torch.ops.load_library()` 在 torch 初始化之后加载，`fork` 出来的子进程要重新加载。构建戳（`_stable_hash.py` 的 `SOURCE_HASH` 与 `.so` 内嵌哈希）不一致时加载直接报错——那是防"跑了旧产物还不自知"，不要绕过。
- **`FLA_NPU_STABLE_LIB` 只用于对照。** 它优先于包内那份，长期开着指向旧 `.so` 会静默使用旧产物。
- **非连续输入如实交出去。** 张量的 sizes / strides / storage offset 原样交给 `aclCreateTensor`，适配层不判布局能力、也不做 dense 拷贝：能不能正确寻址是算子的责任。conv1d 的 `conv_state` 能不能吃 stride 跟着 CANN 走（`CausalConv1d` 的 aclnn 接口是构建期生成的，没有手写 `op_host/op_api`）。在适配层做 staging 会白白付出约 0.1 ms/次的拷贝代价。
- **同名包只能装一个。** `flash-linear-attention-npu` 同名互覆盖，多 SoC / 多版本并存要用独立 venv。
- **构建机的 libstdc++ 水位会跟着产物走。** 薄层与 OPP 的 host 侧库是在构建机上编的；构建机比目标机新时，目标机会在 `import fla_npu` 时报 `GLIBCXX_3.4.x not found`——pip 的 manylinux 标签只承诺 glibc，看不出这条。发布前用 `tools/stable_abi_audit.py --lib` 查一次水位。

## 4. 构建与安装

只构建 Python runtime wheel（在 `torch_custom/fla_npu` 下）：

```bash
python3 setup.py bdist_wheel
```

wheel 里包含 Python runtime、内嵌 OPP 骨架，以及默认编进包内的 `libfla_npu_stable.so`；`FLA_NPU_BUILD_STABLE_ABI=0` 时不编薄层，得到纯 ctypes wheel。安装：

```bash
# WHEEL_PATH 用构建日志里的准确文件名（勿用通配符，避免匹配多个产物）
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
```

随后装算子 run 包即可把真实 OPP 产物合并到同一位置：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu
./build_out/fla_npu_linux-*.run --full
```

覆盖已安装 wheel 内嵌 OPP，或只替换少量算子：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=chunk_fwd_o
./build_out/fla_npu_linux-*.run --full
```

安装器会按算子列出覆盖后的状态（`WARNING` 不可用 / `NOTICE` 需人工关注 / `OK` ABI 一致），覆盖后只保留 `libcust_opapi.so` 并刷新 wheel 的 `RECORD`，重复安装同一 run 包不会重复追加 OPP 路径。

仓库根目录的一键编包（含完整 OPP 与 `libfla_npu_stable.so`）：

```bash
FLA_NPU_SOC=ascend910b python3 scripts/build_wheel.py
python3 scripts/check_packaged_wheel_api.py
```

改过 `csrc/` 后单独重编薄层：

```bash
python3 torch_custom/fla_npu/csrc/build_stable.py --out /tmp/libfla_npu_stable.so --no-debug-probe
```

> 安装流程看护：构建与安装完成后执行 `python scripts/check_install_workflows.py`，CI 也会跑。

## 5. legacy torch_npu / torch.ops.npu 路径

需要兼容 `torch_npu.ops.xxx` 时显式安装 Python wrapper：

```python
import torch_npu
from fla_npu.ops import ascendc

ascendc.install_torch_npu_ops_compat()
torch_npu.ops.npu_xxx(...)
```

需要兼容更旧的 `torch.ops.npu.xxx` 时要显式生成并构建 legacy extension：

```bash
bash gen.sh npu_custom.yaml
FLA_NPU_BUILD_LEGACY_EXTENSION=1 python3 setup.py bdist_wheel
```

```python
import fla_npu

fla_npu.load_legacy_torch_ops()
torch.ops.npu.npu_xxx(...)
```

legacy 路径会生成 `op_plugin/`、`torch_npu/csrc/`、`custom_aclnn_extension_lib*.so` 等产物，并重新绑定 PyTorch / Python / C++ ABI 与 torch_npu dispatcher 行为，因此只用于历史接口兼容或专项验证。`torch.ops.npu.*` / `torch_npu.ops.*` 只支持到 v26.6.0，迁移步骤见[兼容与迁移指南](../../docs/兼容与迁移指南.md)。**新代码不要以 legacy 路径作为唯一调用方式。**

## 6. 测试要求

- 新测试默认用 `from fla_npu.ops import ascendc as ascendc_ops`，不要调用 `fla_npu.load_legacy_torch_ops()`。
- 不要把 `torch.ops.npu.*` 当默认正确性路径；确实要覆盖 legacy 时单独写清目的，并显式打开 `FLA_NPU_BUILD_LEGACY_EXTENSION=1`。
- 需要 NPU 的设备回归放在 `tests/stable_abi/`，跑法与分组见该目录的 README。
