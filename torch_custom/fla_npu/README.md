# fla_npu 适配层

`torch_custom/fla_npu` 把 Ascend C 算子变成可调用的 Python 接口。上层只认一个入口：

```python
from fla_npu.ops.ascendc import chunk_fwd_o

out = chunk_fwd_o(...)
```

短名（不带 `npu_` 前缀）由 `fla_npu/ops/ascendc/__init__.py` 的 `_strip_npu_prefix()` 统一导出。
它**不是适配层引入的**：随“显式加载 legacy torch op”那次改动进的 main，早于适配层；带 `npu_`
前缀的名字（`ascendc.npu_chunk_fwd_o`）也还能导入，但文档与新代码统一用短名。

## 1. 新增算子适配

一次适配 = **新建 1 个算子文件 + 1 行 include + 2 行注册 + 1 个 Python wrapper**。**不需要先写一份 ctypes 适配**：
ctypes 只是回退后端，新算子默认没有它，按 `_LAUNCHER_ONLY_OPS` 声明即可（接入指南 §8）。

### 1.1 交付件

```text
torch_custom/fla_npu/
├── csrc/src/stable_<op>.cpp        # 新建：文件名 = 算子名去掉 npu_，一个算子一个文件
│   └── kSchema_<op> + run_<op>：申请输出 + 一条 FLA_STABLE_EXEC
├── csrc/src/stable_ops.cpp         # 改：include 一行 + 注册两行
│   └── #include "stable_<op>.cpp"; m.def(kSchema_<op>); m.impl("<op>", &boxed_adapter<run_<op>>);
└── fla_npu/ops/ascendc/
    ├── _stable.py                  # 改：加一个真签名 wrapper
    │   └── def <op>(...): return _op("<op>")(..., _current_stream_ptr())
    └── __init__.py                 # 改：下面两种情形按需，都不适用就一行都不用动
        ├── 算子原地写参数 → MUTATED_ARGUMENTS 加一行（必要时 MUTATION_FLAGS）
        └── 没有 ctypes 回退（新算子默认）→ 名字加进 _LAUNCHER_ONLY_OPS
```

公开名不需要加白名单：`_get_stable_op(name)` 就是 `getattr(_stable, name, None)`，有同名函数即走适配层。

### 1.2 每个文件的规范

| 文件 | 规范 |
| --- | --- |
| `stable_<op>.cpp` | **一个算子一个文件，用宏写，不写手写入口**：`kSchema_<op>` 形参 === `run_<op>` 形参 === `FLA_STABLE_EXEC` 实参 === aclnn 头文件顺序（`stream` 固定在最后）；算子私有的名表/helper 也放这里 |
| `stable_<组>_common.cpp` | 只放**被 ≥2 个算子共用**的 helper（当前只有 `stable_causal_conv1d_common.cpp`、`stable_fwd_h_common.cpp`）；新建时在 include 列表里排在用它的算子之前 |
| `stable_ops.cpp` | 两件事：include 各算子文件（common 在前、其余按名字排序）+ 注册两行 |
| `_stable.py` | 真签名 wrapper（不要 `*args` / `**kwargs`），位置参数顺序与 schema 形参一致；字符串用 `_char_code`、host 数组用 `_host_ints`、stream 用 `_current_stream_ptr()` |
| `__init__.py` | 只在「原地写参数」或「没有 ctypes 回退」时才改，其余情形不动 |

### 1.3 完整规范在哪

参数类型对照、模板、文件划分规则、门禁命令、设备回归矩阵和常见坑都在
[`docs/architecture/适配层接入指南.md`](../../docs/architecture/适配层接入指南.md)：
§1 交付件清单、§2 文件划分（一算子一文件）、§3 参数类型对照、§4 模板、§5 落地步骤、§6 设备回归矩阵、§7 常见坑、§8 没有 ctypes 回退的算子。

「为什么必须用宏」的完整理由（boxed kernel 的引用所有权契约、手写入口漏引用导致 191 MiB 泄漏的事故）
见 [`适配层设计.md`](../../docs/architecture/适配层设计.md) §6。

## 2. 现状

**默认后端是 `csrc/` 编出的 Stable-ABI 适配层**（`libfla_npu_stable.so` + `_stable.py`），当前 31 个公开算子全部由它承载，张量走 torch dispatcher。**ctypes**（`_aclnn_ctypes.py`）退居**回退后端**：怀疑适配层有问题时用 `FLA_NPU_STABLE_ABI=ctypes` 走它做对照；新算子不再要求写 ctypes 适配。

适配层不绑 CPython ABI、也不绑 libtorch C++ ABI，一份 wheel 可跨 Python / torch 版本使用（最低已验证 torch 2.7.1）；host 开销与 vLLM-Ascend 的 custom 路径同一量级。

| 路径 | 职责 |
| --- | --- |
| `csrc/include/stable/exec.h` | `FLA_STABLE_EXEC`：参数 holder（保活到 launch 之后）、符号解析、workspace、下发 |
| `csrc/include/stable/boxed.h` | `boxed_adapter<run_*>`：按 `run_*` 的签名拆 boxed 栈、按返回值类型打包 |
| `csrc/include/stable/{acl_meta,runtime,layout_math,at_facade}.h` | 张量元信息与输出分配、workspace / stream、layout（BSND / BNSD / TND / NTD）换算、可用的 torch C API 门面 |
| `csrc/src/stable_<op>.cpp` | 一个算子一个文件：一个算子 = 一条宏（文件划分见 [接入指南 §2](../../docs/architecture/适配层接入指南.md)） |
| `csrc/src/stable_<组>_common.cpp` | 被 ≥2 个算子共用的 helper（当前 2 个：conv1d、fwd_h） |
| `csrc/src/stable_ops.cpp` | 只做注册（`m.def` + `m.impl`），并把各算子文件 include 进同一个编译单元 |
| `fla_npu/ops/ascendc/_stable.py` | 适配层的 Python wrapper（真签名）、后端选择、取 stream |
| `fla_npu/ops/ascendc/_aclnn_ctypes.py` | ctypes 参考实现 |
| `fla_npu/ops/ascendc/__init__.py` | 公开入口、短名导出、正反向绑定、mutation 契约 |
| `tools/*.py` | 离线门禁 |
| `tests/stable_abi/` | 需要 NPU 的设备回归 |

运行期开关（后端切换 / 诊断 / 逃生阀）见[适配层设计 §7](../../docs/architecture/适配层设计.md)：
默认值就是推荐值，正常使用不需要设置任何一条。

## 3. 注意事项

- **stream 每次现取，绝不要缓存。** 进程级缓存过一个 stream pointer，在 vLLM 的多线程多 stream 下会把 kernel 发到别的线程的 stream 上（512 那次崩溃就是这么来的）。适配代码不要自己取 stream，也不要把值存下来。
- **取 stream 的 accessor 要和下发方式配对。** 下发走 torch_npu 任务队列（vLLM `EXEC_NPU_CMD` 同路）时用不排空队列的 accessor，顺序由队列保证；内联直投时必须用会排空队列的那把，否则 kernel 会插到已入队任务的前面。配错在空队列上看不出来，在 vLLM worker 上会变成每次调用约 1 ms。这段逻辑固定在 `_stable.py::_current_stream_ptr()` 里，两个逃生阀是 `FLA_NPU_STABLE_STREAM=accessor` 与 `FLA_NPU_STABLE_LAUNCH=inline`。
- **launcher 的加载时机。** 由 `torch.ops.load_library()` 在 torch 初始化之后加载，`fork` 出来的子进程要重新加载。构建戳（`_stable_hash.py` 的 `SOURCE_HASH` 与 `.so` 内嵌哈希）不一致时加载直接报错——那是防"跑了旧产物还不自知"，不要绕过。
- **非连续输入如实交出去。** 张量的 sizes / strides / storage offset 原样交给 `aclCreateTensor`，适配层不判布局能力、也不做 dense 拷贝：能不能正确寻址是算子的责任。conv1d 的 `conv_state` 能不能吃 stride 跟着 CANN 走（`CausalConv1d` 的 aclnn 接口是构建期生成的，没有手写 `op_host/op_api`）。在适配层做 staging 会白白付出约 0.1 ms/次的拷贝代价。
- **同名包只能装一个。** `flash-linear-attention-npu` 同名互覆盖，多 SoC / 多版本并存要用独立 venv。
- **构建机的 libstdc++ 水位会跟着产物走。** 适配层与 OPP 的 host 侧库是在构建机上编的；构建机比目标机新时，目标机会在 `import fla_npu` 时报 `GLIBCXX_3.4.x not found`——pip 的 manylinux 标签只承诺 glibc，看不出这条。发布前用 `tools/stable_abi_audit.py --lib` 查一次水位。

## 4. 构建与安装

只构建 Python runtime wheel（在 `torch_custom/fla_npu` 下）：

```bash
python3 setup.py bdist_wheel
```

wheel 里包含 Python runtime、内嵌 OPP 骨架，以及默认编进包内的 `libfla_npu_stable.so`；`FLA_NPU_BUILD_STABLE_ABI=0` 时不编适配层，得到纯 ctypes wheel。安装：

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

改过 `csrc/` 后单独重编适配层：

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
