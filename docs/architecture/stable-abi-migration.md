# Thin launcher：torch Stable ABI 迁移方案

> 分支：`feat/stable-abi-thin`（基于 `feat/fla-npu-thin-launcher` / PR #496）
> 目标：把 `_C_thin` 从"pybind11 + `at::Tensor`（libtorch C++ ABI + cpXXX）"迁到
> "dispatcher 注册 + `torch::stable`（C 符号 ABI）"，用一个产物同时覆盖多个
> torch 版本与多个 Python 版本。

## 1. 现状的三条 ABI 轴与本方案的目标

`_C_thin.cpython-311-aarch64-linux-gnu.so` 当前：

| 轴 | 现状 | 证据 |
| --- | --- | --- |
| CPython ABI | 每 Python 版本一个包 | `WHEEL: Tag: cp311-cp311-linux_aarch64` |
| libtorch C++ ABI | 必须与编译时那份 torch 一致 | `readelf -d` NEEDED `libc10.so`/`libtorch_cpu.so`/`libtorch_python.so`；`nm -D` 27 个 `__cxx11` 未定义符号 |
| CANN | 已解耦（运行时 dlopen） | `csrc_thin/src/runtime.cpp` 只用 `dlfcn.h` |

目标：消掉前两条。**torch 本身（张量/分配器/设备）在运行期不可能去掉**——
输入输出必须是 torch 张量，能解的只是"ABI 与版本耦合"。

## 2. 为什么走 dispatcher 注册（而不是 ctypes C ABI）

两者都能消掉两条轴，差别在"谁做 Python↔张量转换"和"op 对编译器是否可见"：

| 入口 | 转换谁做 | 消 Python 轴 | torch.compile/cudagraph 可见 | 备注 |
| --- | --- | --- | --- | --- |
| pybind11 + `at::Tensor`（现状） | pybind 的 torch caster（不稳定） | ✗ | ✗ | 换 torch 必须重编 |
| **dispatcher + `torch::stable`** | **PyTorch 自己**（随运行版本匹配） | ✓ | ✓ | 本方案 |
| ctypes + 自有 C ABI | 我们自己读元数据 | ✓ | ✗ | 老 ctypes 慢的 91% 在 Python 建销 descriptor（0.328 ms）与三段式 FFI（0.177 ms），搬进 C++ 后可行，但编译器看不见 |

`torch.ops` 只是**实现细节**：对外的 `fla_npu.ops.ascendc.*` 不变，legacy
`torch.ops.npu.*` 的废弃计划不变（那是 torch_npu op-plugin 提供的命名空间，
与本方案的私有命名空间 `fla_npu_thin::` 不是一回事）。

## 3. 已核实的 stable ABI 能力（torch 2.7.1 / 2.9 均有）

| 需要 | 接口 | 位置 |
| --- | --- | --- |
| 张量表示 | `torch::stable::Tensor`（opaque `AtenTensorHandle`） | `torch/csrc/stable/tensor.h` |
| 解包/打包 | `to<Tensor>` / `to<std::optional<Tensor>>` / `to<double>` … / `from(...)` | `stable/stableivalue_conversions.h` |
| 注册 | `STABLE_TORCH_LIBRARY(ns, m){m.def(schema)}` + `STABLE_TORCH_LIBRARY_IMPL(ns,k,m){m.impl(name,fn)}` | `stable/library.h` |
| 运行时符号 | `aoti_torch_library_init_def / fragment / impl`（`libtorch_cpu.so` 导出） | `aoti_torch/c/shim.h` |
| 分配 | `aoti_torch_empty_strided`、`stable::ops::empty_like` | 同上 |
| 元数据 | `aoti_torch_get_data_ptr / dim / numel / storage_numel / sizes / strides / dtype / device_type / device_index / storage_offset / is_contiguous` | 同上 |
| 从裸指针建张量 | `aoti_torch_create_tensor_from_blob(_v2)` | 同上 |
| stream | `stable::accelerator::getCurrentStream(device_index)` / `getCurrentDeviceIndex()` / `aoti_torch_get_current_cuda_stream` | `stable/accelerator.h` |
| ABI 兼容策略 | `shim.h` 明文：改动必须保持 ABI，加参数要出 `_v2` 并保留旧版；`aoti_torch_abi_version()` 可探测 | `aoti_torch/c/shim.h` |

## 4. 阶段与门禁

### Phase 0：清点与静态门禁（离线，零风险）

- 产出 `torch API → stable 等价物` 对照表（本文档第 3 节是起点）。
- 新增 `tools/stable_abi_audit.py`：
  1. 源码级：stable 构建的源文件不得出现 `ATen/`、`c10/`、`pybind11`、`torch/extension.h`；
  2. **ELF 级（关键）**：`nm -D --undefined-only libfla_npu_thin.so` 中不得出现
     C++ mangled 的 ATen/c10 符号（`_ZN2at*`、`_ZN3c10*`），只允许
     `aoti_torch_*` + libc/libstdc++。这条把"是否真的只用稳定符号"变成可测断言。
- 门禁：audit 对现状 `.so` 必须报违规（证明它能拦），对新 `.so` 必须通过。

### Phase 1：单算子竖切（本分支已开始）

算子顺序：
1. `npu_recurrent_gated_delta_rule`——非连续 paged state、可选张量、原地写 state、
   stream 语义，且已有完整 host 基准与 parity 资产；
2. 第二个算子：`npu_chunk_kda_fwd`（11 输出 + flag 驱动可选输出，验证多返回与
   codegen 可行性）；若风险偏高则先用 `npu_recurrent_kda`（同族、双输出）。

测试套件（复用为主）：

| # | 测试 | 复用的资产 | 通过标准 |
| --- | --- | --- | --- |
| T1 | 数值 parity（ctypes vs stable） | `tests/regression_thin_ops.py` 的 GDR 场景输入构造 | 逐输出 diff 0.0，state 亦 0.0 |
| T2 | mutation 契约 | `tests/regression_mutation_contract.py` | 该 bump 的 bump、requires_grad 拒绝行为一致 |
| T3 | 多线程多 stream | `test_thin_stream_interleaving.py` 的用法 | 事件落在调用线程的 stream |
| T4 | 非连续 state（gap/offset） | 同上 GDR 场景 | parity 0.0 |
| T5 | 三后端 host A/B | `bench_recurrent.py` / `decompose_recurrent.py` | stable ≤ pybind × 1.15 |
| T6 | ELF 符号审计 | 新增 `tools/stable_abi_audit.py` | 无 `_ZN2at/_ZN3c10` 未定义符号 |
| T7 | 非法输入 | 现有兼容性测试 | 只要求"会报错"，不要求同型 |

门禁 G1：T1/T2/T3/T6 全绿 + T5 达标，否则先定位再决定是否继续。

### Phase 2：跨 torch 版本加载矩阵（收益证明）

用**低版本** torch（2.7.1）编一次 `libfla_npu_thin.so`，同一产物在下列环境跑 T1/T3/T6：

- 241：torch 2.7.1 + torch_npu 2.7.1（py3.10，conda fzy）
- 241：torch 2.9 + torch_npu 2.9（py3.12，系统 python）
- 221：torch 2.9.0 + torch_npu 2.9.0.post2（py3.11）

反向验证（用 2.9 编、在 2.7 加载）记录失败形态 → 得出"按最低支持版本编译"规则。
门禁 G2：同一产物在 ≥2 个 torch 版本上 parity 全绿。

### Phase 3：codegen 后端 + 全量迁移

给 `tools/op_spec_codegen.py` 增加 `stable` 后端：同一份 spec 生成
（a）`StableIValue` 解包/打包、（b）`STABLE_TORCH_LIBRARY(_IMPL)` 注册、
（c）schema 字符串（可复用 `npu_custom.yaml` 里的 `Tensor?` / `Tensor(a!)`）。
分批迁移（A：recurrent/kda；B：chunk fwd/bwd 家族；C：conv1d + A5-only），
每批都要在 910b 与 950 上跑完对应 parity 矩阵。

### Phase 4：打包与切换

- 产物 `libfla_npu_thin.so`（无 cpXXX）；wheel 平台标签改为不依赖 Python 版本；
- 元数据写 `torch>=<实测下限>,<上限`；import 时用 `aoti_torch_abi_version()` 做一次检查；
- `FLA_NPU_THIN_ABI=stable|pybind|ctypes` 三门并存，默认切换只在 G2/G3 全绿后进行。

## 5. 风险与退路

| 风险 | 验证点 | 退路 |
| --- | --- | --- |
| dispatcher 开销高于预期 | T5 | 改 C ABI + ctypes 入口（归因数据已给出上限） |
| `Tensor(a!)` 的 version/grad 语义与手写 wrapper 不一致 | T2 | 保留 `MUTATION_FLAGS` wrapper |
| stable stream 未落到 NPU | 先显式传 stream（Python raw accessor），再单独验证 `getCurrentStream` | 长期保留显式 stream 参数 |
| 未定义符号解析失败（load_library 作用域） | T6 + 首次加载 | 让产物 DT_NEEDED 指向 `libtorch_cpu.so`（同 SONAME 已加载，由 loader 复用） |
| 某算子需要 stable 表面没提供的能力 | Phase 0 对照表 / 每批 audit | 该算子留在 pybind 后端，双后端同源共存 |

## 6. 本分支的当前进度

- `csrc_stable/src/stable_recurrent_gdr.cpp`：recurrent GDR 的 stable 注册 + 实现骨架
  （复用 `csrc_thin/src/runtime.cpp` 的 dlopen 符号解析，descriptor 语义与 ctypes 对齐）。
- `csrc_stable/build_stable.py`：不链接 torch 编译期头之外的任何东西（只 include
  `torch/csrc/stable/*`），产出 `libfla_npu_thin.so`。
- `tests/regression_stable_abi.py`：T1/T2/T5/T6 的驱动（T3/T7 在 Phase 1 收尾补）。
