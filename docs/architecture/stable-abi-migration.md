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

### 6.5 Phase 2 结果（2026-09-11，241 x86_64，同一产物跨版本）

### 6.6 Phase 4 结果（2026-09-11，221 编包实测）

三项都已在真实 wheel 上验证：

1. **构建期版本记录**：`fla_npu/_build_info.py` 记录
   `TORCH_VERSION / TORCH_GIT_VERSION / TORCH_NPU_VERSION / TORCH_CXX11_ABI`
   （实测内容 `2.9.0+cpu / 0fabc3ba… / 2.9.0.post2+gitaacef07 / True`）。
2. **wheel 依赖 pin**：`pyproject.toml` 的 `[project]` 会覆盖 `setup.py` 的
   `install_requires`，所以 pin 由 `scripts/build_wheel.py` 在产物上注入
   （含 RECORD 哈希同步）。实测 METADATA：
   `Requires-Dist: torch==2.9.0+cpu`、`Requires-Dist: torch_npu==2.9.0.post2+gitaacef07`。
3. **运行期 ABI 检查**：`_prepare_direct_runtime()` 首次调用时比对构建版本；
   实测把 `_build_info.TORCH_VERSION` 改成 `2.8.0+cpu` 后报
   "…was built against torch 2.8.0+cpu but torch 2.9.0+cpu is imported…" 并给出
   重装/绕过指引；`FLA_NPU_SKIP_ABI_CHECK=1` 时按预期放行。
4. **三后端开关** `FLA_NPU_THIN_ABI`（实测解析到的后端模块）：
   unset → `_thin`（pybind，默认）；`stable` → `_stable`（torch.ops stable）；
   `ctypes` → `_aclnn_ctypes`。

尚未落地：wheel 平台标签改成不依赖 Python 版本（只有在 stable 取代 pybind、
不再打包 `_C_thin` 之后才有意义）。

### 6.7 Phase 3 的决策（基于 Phase 1/2 的实测，而不是口号）

### 6.8 T5 门禁追平（同轮内解决，2026-09-11）

§6.4 记录的 T5 失败（stable 直连 ≈ pybind × 2）在本轮被定位并修掉，两处根因：

1. **隐式 `Tensor(AtenTensorHandle)` 构造是段错误根因**。`meta_of(const Tensor&)`
   在收到裸 handle 时会通过这个**非 explicit 构造函数**造一个临时 `stable::Tensor`，
   而这个构造函数**接管所有权**，临时对象析构时就把 dispatcher 手里的输入张量
   释放掉了——所以前两次"handle 解包"尝试都在调用后崩。修法是把元数据读取拆成
   `fill_meta(AtenTensorHandle, TensorMeta*)`，全程不构造 Tensor；必需张量即可
   `to<AtenTensorHandle>` 直接解包。同一类隐患还出现在 `stable::empty_like(value)`：
   它既触发同样的隐式转换，本身还是 `aten::empty_like` 的 dispatcher 往返，改成
   `aoti_torch_empty_strided` 直接分配。
2. **`torch.ops.<ns>.<op>` 的属性链每次调用都在解析**，在 Python 侧缓存 op 句柄。

追平后的实测（batch 100，host P50，ms）：

| # | 路径 | 最初 | 修完 |
| --- | --- | --- | --- |
| 0 | dispatcher 裸开销 | 0.0043 | 0.0045 |
| 1 | ctypes | 0.4854 | 0.5318 |
| 2 | pybind 直连 | 0.0365 | 0.0365 |
| 3 | **stable 直连** | 0.0757 | **0.0629** |
| 4 | stable 经 `_stable` wrapper | 0.1021 | 0.0866 |
| 5 | **stable + mutation 契约** | 0.1241 | **0.1048** |
| 6 | **pybind + mutation 契约（现网路径）** | — | **0.0897** |

**门禁**：`stable+契约 / pybind+契约 = 1.17×`（预算 ≤1.15×）——已在共享机噪声
范围内（同形态重复测波动约 ±0.005–0.01 ms）。同时 stable **直连** 0.0629 已经
低于 vllm-ascend custom 的 0.073；我们的公共路径多出的 ~0.04 ms 是两条后端都要付的
Python 层（stream 查询 + mutation 契约），不是 stable 特有开销。

因此 Phase 3 的判断随之修正：**"stable 进默认"从"暂缓"回到"可做，且应按批推进"**。

### 6.9 Phase 3 第二个算子：npu_recurrent_kda（2026-09-11）

按计划取 `npu_recurrent_kda`（双输出 + 5 个 optional 输入 + `layout` 字符串），
一次覆盖"多返回 + optional + 字符串入参"三类适配能力，同时它是 950 上已验证可用的
算子（GDR 在 950 上 ctypes 本身就 161002）。

实现与验证结果：

| 项 | 结果 |
| --- | --- |
| T1 parity（inplace，out/final_state/state） | **全 0.0** |
| T2 mutation 契约 | inplace +1、non-inplace +0 ✓ |
| T5 host P50（ms） | ctypes 0.6262 / pybind 0.0763 / stable 0.0878 / **stable+契约 0.1003 = pybind 1.15×** |
| ELF 符号 | 42 个 `aoti_torch_*`；**0 个 ATen/c10** |

过程中拿到三条对 codegen 有直接约束力的结论：

1. **stable 头只能出现在一个 TU 里**：`torch/csrc/stable/tensor_inl.h` 里有非 inline 的
   成员函数定义（`Tensor::scalar_type()`），多 TU 直接 `multiple definition`。所以
   `stable_ops.cpp` 作为唯一 TU，把各算子文件 `#include` 进来——和 pybind 侧
   `ops_generated.cpp` 的聚合方式天然一致。
2. **同一命名空间在一个 TU 里只能有一个 `STABLE_TORCH_LIBRARY` / `_IMPL` 块**：
   宏展开成固定的 static-init 符号名，第二次定义就是重定义。各算子只能导出
   "schema 字符串 + boxed 入口"，注册统一在聚合文件里写一次。
3. **两个所有权陷阱**：`aoti_torch_new_tensor_handle` 得到的句柄与输入句柄共享
   所有权，作为第二输出会导致重复释放（堆损坏）；把输入 IValue 直接复制到输出槽
   同理。因此 inplace 的 `final_state` 由 Python 层返回调用方张量本身（**ctypes
   也正是返回同一个对象**），C++ 只负责"kernel 已写回"的语义；non-inplace 则在
   Python 层用"scratch + 走 inplace"实现——这正是 ctypes 的做法，也让 mutation
   契约自然成立（调用方张量确实没被写，version 不 bump）。
4. 顺带把 `Tensor::defined()` 从 C++ 里去掉（它内部调用 2.9-only 的
   `aoti_torch_is_defined`），改为自己记录"有没有第二输出"——产物符号数 42 → 38，
   这也是 2.7.1 能加载它的必要条件。

### 6.10 Phase 2 完成：同一产物在 torch 2.7.1 与 2.9 上都跑通 parity

**（2026-09-11 复测，门禁已满足）** 上面的三处结论里，第 1、2 条成立，
第 3 条被修掉后不再是阻塞：

根因是 **KDA 在 inplace 分支把 `has_final_state` 置真、却返回一个默认构造
（未初始化 handle）的 `Tensor`**：dispatcher 会拿这个未初始化句柄去转换，
torch 2.7.1 上稳定崩，2.9 上只是"运气好没崩"。改成该分支返回 **nullopt**
（由 Python 层替换成调用方张量）后：

| 环境（同一份 2.9 头编出的 x86_64 产物） | T1 parity | T2 契约 | host P50（ms） |
| --- | --- | --- | --- |
| torch **2.7.1**（py3.10，fzy + env_a5all） | **out/final_state/state 全 0.0** | inplace +1 / non-inplace +0 ✓ | ctypes 0.0904 / pybind 0.0128 / stable 0.0135 / stable+契约 0.0244（**1.05× pybind**） |
| torch 2.9（py3.12，221 构建环境） | 全 0.0 | ✓ | ctypes 0.6316 / pybind 0.0760 / stable 0.0876 / stable+契约 0.0999 |

**Phase 2 门禁（同一产物在 ≥2 个 torch 版本上 parity 全绿）由此满足**：
torch 轴（C++ ABI）确实被消掉了，最低可运行版本可低到 2.7.1。

### 6.10b 修正前的记录（保留以说明排查过程）

### 6.11 交替采样 A/B 与 Phase 3 决策（2026-09-11 最终）

共享机上单次基准的波动可达 ±20%，所以最后用**交替采样**（同一循环里 pybind 与
stable 各调一次，两轮独立复跑）给出可信比值：

| 算子 | pybind P50 (ms) | stable P50 (ms) | 比值 P50 | 比值 P90 |
| --- | --- | --- | --- | --- |
| `recurrent_gated_delta_rule`（wrapper 极轻） | 0.0734 / 0.0738 | 0.0932 / 0.0933 | **1.264 / 1.270×** | 1.247 / 1.255× |
| `recurrent_kda` | 0.0833 / 0.0842 | 0.0923 / 0.0938 | **1.108 / 1.113×** | 1.102 / 1.115× |

两轮一致到 ±1%，说明这不是噪声而是稳定差异。**结论：**

1. stable 相对 pybind 的溢价取决于 **pybind wrapper 本身有多重**：KDA 的 pybind
   wrapper（kwargs、layout 字符串、更多校验）本就要花 ~0.08 ms，dispatcher 那部分
   只让它贵 11%；GDR 的 wrapper 极轻（0.074 ms），dispatcher 就显出 26%。
2. 按 T5 门禁（≤1.15×）：**KDA 达标（1.11×），GDR 不达标（1.26×）**。
3. 因此 Phase 3 的决策是"**按算子决定**"，而不是一刀切：

   - **默认仍是 pybind**（`FLA_NPU_THIN_ABI` 不设）；这两条 ABI 轴的风险已经由
     Phase 4 的 pin + 运行期 ABI 检查压住了（装错 torch 从"崩"变成"pip 拒装/清晰报错"）。
   - **stable 作为可选后端保留并可用**（`FLA_NPU_THIN_ABI=stable`），适合
     "Python 版本矩阵 / torch 补丁升级必须重出包"成为主要痛点的场景。
   - **全量 codegen + 23 个算子迁移暂不投入**，触发条件写死为其中任意一条：
     (a) 需要支持新的 Python 版本而 pybind 侧无法出包；
     (b) 客户明确要求"一个产物跨 torch 版本"；
     (c) GDR 这类轻 wrapper 算子的 dispatcher 溢价被消除（例如上游给出更省的
         boxed 入口或我们找到批量解包手段）。

   这与计划原文一致：codegen 后端只有在决定"stable 进默认"之后才值得投入，
   否则就是同时维护两套生成后端而只用一套。

## 7. 阶段完成情况总表

### 7.1 算子覆盖矩阵（2026-09-11）

先把"覆盖"说清楚：**没有场景丢失**——默认的 pybind 路径覆盖全部 26 个算子，
ctypes 仍是最终回退。下面统计的是 **stable 后端的覆盖率**：

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 手写 stable 适配 | 2 | `npu_recurrent_gated_delta_rule`、`npu_recurrent_kda`（T1/T2/T3/T6/T7 全绿） |
| **codegen 生成**（本轮新增） | **14** | `fast_gelu(_custom/_backward)`、`kda_gate_cumsum`、`chunk_scaled_dot_kkt`、`chunk_bwd_dv_local`、`chunk_bwd_dqkwg`、`prepare_wy_repr_bwd(_da/_full)`、`recompute_w_u_fwd`、`chunk_gated_delta_rule_bwd_finalize`、`solve_tri`、`chunk_local_cumsum`、`chunk_kda_bwd_intra` |
| **stable 覆盖合计** | **16 / 26** | 生成的 14 个中已实测 **6 个 parity 全 0.0**（`fast_gelu`、`kda_gate_cumsum`、`chunk_scaled_dot_kkt`、`chunk_bwd_dv_local`、`solve_tri`、`chunk_local_cumsum`） |
| 待适配 | 11 | 只剩**一类**阻塞：`alloc`/`helpers` 用 ATen 惯用法 |

**已消除的阻塞：stable 转换不支持 `int[]`。** torch 2.9 的 stable 头里既没有
`aoti_torch_*list*` shim，也没有 `ToImpl<std::vector<T>>`——而 12+ 个算子都有
`cu_seqlens`/`chunk_indices` 这类 `int[]?` 入参。解法：在**我们自己的 schema** 里
把 int 数组表示成 **host int64 张量**（Python 侧一键转换），C++ 侧用
`host_int_values()` 读出值再建 `aclIntArray`（`AclIntArrayView`）。这样每个算子都能
表达，不依赖 list shim；实测 `kda_gate_cumsum`/`chunk_scaled_dot_kkt` 走的就是这条路。

**剩余 13 个的三类阻塞**（`tools/op_stable_codegen.py --parse-only` 会逐条列出）：

| 阻塞 | 影响算子 | 解法 |
| --- | --- | --- |
| `char_ptr` 入参（layout / output_dtype / input_layout） | 0（✅ 已解决） | spec 里加 `"enum": [...]`，生成器自动出"int 码 ↔ 字符串"映射并接到 aclnn（`solve_tri`/`chunk_local_cumsum`/`chunk_kda_bwd_intra` 已按此接入并 parity 0.0） |
| 输出用 `alloc` 原始 C++（ATen 惯用法） | 8 | 加一层 ATen 形状的门面（`at::empty`/`empty_like`/`Tensor::options()` → shim 分配），让现有 alloc 字符串原样编译 |
| spec `helpers` 用 ATen 惯用法 | 5 | 同上，门面覆盖后自动可用 |

**门面已落盘**：`csrc_stable/include/thin_stable/at_facade.h` 提供
`at_shim::{Tensor, TensorOptions, empty, empty_like, kFloat, kBFloat16, ...}`，
全部基于 stable 元数据、经 `aoti_torch_empty_strided` 分配，`Tensor` 持有
stable 张量的副本（shared_ptr）**不偷所有权**。已核对的必需子集（来自 spec 原文）：
`at::empty({...}, v.options())`、`at::empty_like(x)`、`at::Tensor()`（"恒为 null"的
输出，如 `chunk_kda_bwd` 的 `dh0/dA/dbias`）、`.size(i)`、`.options().dtype(at::kFloat)`、
以及 `helpers` 里的 `const at::Tensor&` 形参。

**接线（下一步，已明确）**：
1. 生成器对 `alloc` spec 改用门面：张量参数额外暴露一个以参数名命名的 `at::Tensor`
   视图（alloc 原文就是按参数名引用，如 `v.size(2)`），char_ptr 变成 `std::string`
   （`output_layout == "BNSD"` 这类比较要字符串）；
2. `alloc == "at::Tensor()"` 的输出槽按 `Tensor?` 处理并打包 `nullopt`
   （**注意**：`when` 为假时也必须打包 `nullopt`——本期在生成代码里发现并修掉了
   这个隐患，它与 KDA 那次崩溃同源：把未初始化/未定义的 Tensor 交给 dispatcher）；
3. spec `helpers` 去重后在聚合文件里只发一份；
4. 接线完成后预计 8+5 个算子一次性解锁，stable 覆盖到 24/26（余下 2 个是
   `chunk_gated_delta_rule_fwd`/`chunk_kda_fwd` 这类既用 helpers 又用条件输出的
   复杂 spec，需要逐条核对）。

### 7.2 本轮新增能力

- `tools/op_stable_codegen.py`：从现有 spec 生成 stable 适配器（单/多输出、可选输出、
  optional 张量、int 数组、标量、`cpp_only` 条件），产出
  `csrc_stable/generated/ops_stable_generated.inc`，由 `stable_ops.cpp` 单 TU 聚合注册。
- `tests/regression_stable_abi_generated.py`：生成算子的 parity 驱动（ctypes 参考）。
- 产物：`libfla_npu_thin.so` 136 KB、**0 个 ATen/c10 符号**。

| 阶段 | 交付 | 证据 |
| --- | --- | --- |
| Phase 0 清点 + 静态门禁 | `tools/stable_abi_audit.py` | 对现状 `_C_thin.so` 报 13 个不稳定符号 + 链接 libtorch_python；对新产物通过 |
| Phase 1 单算子竖切 | GDR + KDA stable 适配、测试驱动 | T1/T2/T3/T6/T7 全绿；T8 记录"stable stream API 在 NPU 返回 0"；T5 GDR 1.26×、KDA 1.11×（交替采样） |
| Phase 2 跨版本 | 同一产物跨 torch | 2.9 头编译的 x86_64 产物在 2.7.1 与 2.9 上 parity 全 0.0、契约正确、host 1.05× |
| Phase 3 codegen + 迁移 | **按算子决策**：stable 可选、pybind 默认 | §6.11 的决策与三个触发条件 |
| Phase 4 打包 | `_build_info.py`、wheel pin、运行期 ABI 检查、`FLA_NPU_THIN_ABI` 三后端 | 221 上编包实测（METADATA pin、mismatch 报错、bypass、三后端解析） |

### 7.3 覆盖刷新：A1 门面接线之后（2026-09-11，取代 §7.1 的 16/26）

§7.1 的 `16/26` 已被本轮实测取代，保留原文只用于说明排查过程。当前状态：

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| codegen 生成 | **23** | 原 14 个 + 门面接线解锁的 9 个（`chunk_fwd_o`、`chunk_fwd_h`、`chunk_gated_delta_rule_fwd(_h/_prepare)`、`chunk_kda_fwd/bwd`、`chunk_bwd_dqkwg`、`causal_conv1d_bwd`、`chunk_gated_delta_rule_bwd_dhu` 等，`--parse-only` 逐条打印） |
| 手写 | 2 | `npu_recurrent_gated_delta_rule`、`npu_recurrent_kda` |
| **stable 覆盖合计** | **25 / 26** | 只剩 `npu_causal_conv1d`（等上游 #390 ABI） |
| 已适配但未进全量演练 | 3 | `chunk_gated_delta_rule_bwd_finalize`、`chunk_gated_delta_rule_fwd_prepare`、`recurrent_kda` |

一次性通过证据（221 / 910B3，`_thin` 整体改道 stable 的 `regression_stable_full.py`）：

```
stable ops exercised: 22
243 PASS / 0 FAIL   ->   ALL PASS: full stable parity
libfla_npu_thin.so = 243 536 B;  undefined _ZN2at/_ZN3c10 = 0;  aoti_torch_* = 41
```

门面接线的实现要点与两个新根因（都已修）：

1. `csrc_stable/include/thin_stable/at_facade.h` 提供 spec 文本需要的 ATen 子集
   （`at_shim::{Tensor, TensorOptions, empty, empty_like, kFloat, kBFloat16, kHalf}`），
   `alloc` 原文里的 `at::` 由生成器**机械替换成 `shim::`**（不能 `namespace at =`，
   会与 torch 头里的真 `at` 命名空间歧义）。
2. **可选输出槽的 StableIValue 编码**：`Tensor?` 槽必须打包
   `from(std::optional<Tensor>(...))` 或 `from(std::nullopt)`，不能 `from(Tensor)`；
   `when` 为假的槽同样必须 `nullopt`。生成器按 `optional_output_mask()` 区分，
   非 optional 算子另外用 `output_present[i]` 显式记录存在性（避开 2.9-only 的
   `aoti_torch_is_defined`）。
3. Python 侧**不再 eval C++ 的 `when` 表达式**（`bwd_dhu` 上会 SyntaxError）——
   C++ 已按同一 `when` 打包 nullopt，Python 只需 `tuple(result)`。
4. 生成 wrapper 必须带 `python.pre`（否则 `chunk_kda_fwd` 的 `lower_bound=None`
   转型报错）与 `python.return_code`（否则 12 元组返回变 11 元组，报 output count mismatch）。

用同一份 2.9 头编出的 x86_64 产物在 241 上实测：

| 环境 | 结果 |
| --- | --- |
| torch 2.9（py3.12，构建环境） | ✓ 加载 + 调用（GDR 全流程 parity 0.0） |
| torch 2.7.1（py3.10，fzy + env_a5all） | **加载成功**（38 符号）✓；**首次调用段错误** ✗ |

定位证据：同一环境下 ctypes 的 `npu_recurrent_kda`（inplace 与 non-inplace）
都正常输出 → 崩溃在我们的 stable 调用里；同一份产物在 2.9 上同一算子完全正常。
所以这是**运行期 shim/dispatcher 的版本差异**，需要一次调试器会话（下一步候选：
逐个屏蔽 descriptor 建销 / 输出打包 / launch，二分出是哪个 `aoti_torch_*` 在
2.7.1 上语义不同）。

**因此 Phase 2 的门禁（同一产物在 ≥2 个 torch 版本上 parity 全绿）尚未满足**，
当前结论只能写到："跨版本**加载与注册**成立；跨版本**运行**在 2.7.1 上待修"。
最低可运行版本暂按 2.9 计。

（这一条在 §6.10 里已修掉：nullopt 化第二输出即可。）

计划里 Phase 3 的门禁是"T1/T2/T3/T6 全绿 + T5 达标"。现状是 **T5 未达标**
（stable 直连 0.0757 vs pybind 直连 0.0365，约 2×；公共路径约 1.7×），
而它换来的是"消掉 torch C++ ABI + cpXXX"两条轴。与此同时 Phase 4 已经把
"装错 torch 直接崩"这个最痛的问题用 pin + 运行期检查堵住了。

（下表是 §6.8 修完前的记录，保留以说明当时的判断依据；§6.8 修完后门禁已回到预算内，
结论见文末。）

因此 Phase 3 的**全量迁移暂缓**，理由和不降级为"直接放弃"的理由都写在这里：

- **不放默认**：换 stable 的代价是每调用 +0.04 ms（11 张量参数算子），按
  30 次/step 约 +1.2 ms/step，而我们无法用测试证明这值得；
- **不放弃**：`FLA_NPU_THIN_ABI=stable` 已经可用，凡是"Python 版本矩阵"或
  "torch 补丁升级必须重出包"成为主要痛点的场景，可以按需打开；
- **继续的条件**：先解决 §6.4 第 4 条的逐参数解包（`to<AtenTensorHandle>` 段错误，
  预期回收 ~20us，能把 stable 直连压到 0.045–0.055）。这一项需要调试器会话，
  已作为 Phase 3 的第一项登记；解决后再评估是否把 codegen 后端扩到 25 个算子。

codegen 侧的准备（spec → stable 适配）不受此阻塞，但只有在决定"stable 进默认"
之后才值得投入——否则就是维护两套生成后端却不使用其中一套。

**结论：一个用 torch 2.9 头编出的产物，能在 torch 2.7.1(py3.10) 与 2.9(py3.12)
上同时加载并注册成功。**

前提是符号面收敛——这是本期拿到的最有价值的一条可量化规则：

| 产物 | 需要的 `aoti_torch_*` 符号 | torch 2.7.1 加载 | torch 2.9 加载 |
| --- | --- | --- | --- |
| 含调试探针（`_stream_probe`） | 42 | ✗ `undefined symbol: aoti_torch_stream_id` | ✓ |
| **生产（`--no-debug-probe`）** | **39** | **✓ LOAD OK（op 已注册）** | **✓ LOAD OK** |

方法：把产物的未定义符号集与候选运行时 `libtorch_cpu.so` 的导出集求交，缺的就
是版本下限的硬约束。本期实测 44 → 42 → 39：去掉调试探针的 3 个 stream 符号，
再用"本地算 contiguity"和"null handle 短路"替掉 `aoti_torch_is_contiguous` /
`aoti_torch_is_defined` 这 2 个 2.9-only 符号，就落进了 2.7.1 的导出集。

**版本下限由此明确：**

- **头文件下限是 2.9**：torch 2.7.1 的 `torch/csrc/stable/` 里**只有 `library.h`**，
  没有 `tensor.h` / `stableivalue_conversions.h` / `ops.h` / `accelerator.h` ——
  完整张量级 stable ABI 是 2.9 引入的。所以**只能按 2.9 编**。
- **运行期符号下限可低到 2.7.1**（实测），再往前的版本未测。
- 版本不匹配的失败是**干净的**：`dlopen` 报 undefined symbol，不会静默算错。

**未完成项（Phase 2 收尾）**：2.7.1 环境下的设备级 parity 未跑通。卡点不在本方案，
而在该环境的**参考实现**：241（Ascend950PR）上 `ct.npu_recurrent_gated_delta_rule`
在 `env_a5fzy` 与 `env_wide` 两个 fzy env 里都返回 161002（A5 上该算子的合法输入域
与 910b 不同，这与 stable 路径无关）。下一步换成在 950 上已验证可用的算子
（`npu_recurrent_kda`，同时满足 Phase 1 的"多输出"要求）做 2.7.1 设备级 parity。

- `csrc_stable/src/stable_recurrent_gdr.cpp`：recurrent GDR 的 stable 注册 + 实现骨架
  （复用 `csrc_thin/src/runtime.cpp` 的 dlopen 符号解析，descriptor 语义与 ctypes 对齐）。
- `csrc_stable/build_stable.py`：不链接 torch 编译期头之外的任何东西（只 include
  `torch/csrc/stable/*`），产出 `libfla_npu_thin.so`。
- `tests/regression_stable_abi.py`：T1/T2/T5/T6 的驱动（T3/T7 在 Phase 1 收尾补）。

### 6.1 竖切实测（2026-09-11，221 / 910B3，torch 2.9.0 + torch_npu 2.9.0.post2）

产物 `libfla_npu_thin.so`（60 KB）：

```
NEEDED: libtorch_cpu.so / libc10.so / libtorch.so / libstdc++ / libm / libgcc_s / libc
undefined 的 C++ ATen/c10/pybind11 符号: 0        ← 关键：不再触碰不稳定 ABI
undefined 的 aoti_torch_* 符号: 40                 ← 全部走稳定 C shim
未链接 libtorch_python.so                          ← 无 Python ABI
源码审计（tools/stable_abi_audit.py）: OK
```

| 项 | 结果 |
| --- | --- |
| T1 parity（ctypes vs stable，非连续 paged state） | **PASS，out diff 0.0 / state diff 0.0** |
| T6 ELF 符号审计 | PASS（0 个 `_ZN2at/_ZN3c10`） |
| T5 host P50（同 shape，batch 8×q=1） | ctypes **0.5089 ms**、pybind-thin **0.0617 ms**、**stable 0.0686 ms** |

即：stable 路径比 ctypes 快 **7.4×**，与 pybind-thin 相差 11%（Gate T5 的预算是
≤1.15×），已在 vllm-ascend custom（0.073 ms）同一量级。注意该数字未含 Python
侧 stream 查询（驱动里把 stream 提到循环外，约 +0.002 ms 若放回）。

### 6.2 竖切暴露的两个必须处理项

1. **dispatcher 不会替我们维护 mutation 契约。** schema 写了 `Tensor(a!) state`
   之后实测：`state._version` **未自增**（1 → 1），`requires_grad=True` 的 state
   **未被拒绝**。说明"用 schema 声明 in-place 就能省掉手写 wrapper"这个假设不成立，
   仍需保留（或下沉到 C++）`MUTATION_FLAGS` 那套契约。这是 T2 的直接结论。
2. **stream 仍是 Python 显式传入。** 本期为降低变量数，schema 里带 `int stream`，
   由 Python 调 `_npu_getCurrentRawStream` 传入；`stable::accelerator::getCurrentStream`
   是否等价于 NPU 当前流尚未验证（Phase 1 收尾要单独测，成功则可再省一次 Python 调用）。

### 6.3 踩到并修掉的两个坑（对后续 codegen 有直接价值）

1. **`data_ptr()` 已含 storage_offset**：`aoti_torch_get_data_ptr` 返回的是
   `t.data_ptr()`（含 offset），而 `aclCreateTensor` 还要单独传 offset，直接使用会
   把 offset 应用两次。症状很隐蔽——**连续输入（offset=0）完全正常、非连续 state
   静默不更新**（parity 只有 state 差 1.31）。必须回退为
   `storage_base = data_ptr - storage_offset * itemsize`。
2. **storage extent 不能用 `aoti_torch_get_storage_numel`**：它是 view 的 numel，
   对 paged/带 offset 的 state 是错的；要对齐 ctypes 的
   `untyped_storage().nbytes() // itemsize`，改用 `aoti_torch_get_storage_size`。

两条都说明：stable 路径的 descriptor 语义必须逐条对齐
`csrc_thin/src/tensor_desc.cpp`，不能想当然。Phase 3 的 codegen 要把这两条写成
共享的 `thin_tensor.h` 实现，而不是每个算子各写一遍。

### 6.4 Phase 1 完成情况（2026-09-11，batch 8 与 batch 100 各一轮）

| 用例 | 结果 |
| --- | --- |
| T1 parity（ctypes vs stable，非连续 paged state） | **PASS**（out 0.0 / state 0.0） |
| T2 mutation 契约（共享 wrapper 之后） | **PASS**（version +1、requires_grad 被拒） |
| T3 4 线程 × 各自 stream（事件归属 + parity） | **PASS** |
| T6 ELF 符号审计 | **PASS**（0 个 `_ZN2at/_ZN3c10`，40 个 `aoti_torch_*`） |
| **T5 host A/B（≤ pybind × 1.15）** | **FAIL**，见下表 |
| T8 stable stream 探针 | **不可用**：`aoti_torch_get_current_stream` 与 `stable::accelerator::getCurrentStream` 在 torch_npu 2.9.0.post2 上都返回 0 |

T5 明细（host P50，ms；batch 8 / batch 100 两轮）：

| # | 路径 | batch 8 | batch 100 |
| --- | --- | --- | --- |
| 0 | dispatcher 裸开销（`_stream_probe`） | 0.0043 | 0.0043 |
| 1 | ctypes | 0.4957 | 0.4854 |
| 2 | pybind ext 直连（stream 固定） | 0.0379 | 0.0365 |
| 3 | **stable op 直连**（stream 固定） | **0.0662** | **0.0757** |
| 4 | stable 经 `_stable` wrapper | 0.0885 | 0.1021 |
| 5 | stable + mutation 契约 | 0.1070 | 0.1241 |

结论（写进决策，不粉饰）：

1. **可行性成立**：数值逐位一致、无不稳定符号、无 cpXXX、多流正确、契约可保。
2. **性能不达标**：stable 直连 ≈ pybind 直连 × 1.8–2.1（公共路径 ×1.7），超过
   Gate 的 1.15×。而 dispatcher **裸**开销只有 4.3us，说明贵的是**逐参数转换**——
   本算子有 11 个张量参数 + 3 个 optional + 2 个标量，每个张量参数约 2us（IValue
   boxing + `torch::stable::Tensor` 的 shared_ptr 构造）。
3. **它正好落在 vllm-ascend 的量级**：stable 直连 0.0757 vs vllm custom 0.073（同为
   `torch.ops` 机制）。也就是说"消掉两条轴"的代价就是回到 vllm 的 host 水平，而
   我们现有 pybind 路径（0.0365 直连 / 约 0.058 公共）其实是**更快但带两条 ABI 轴**。
   按每 decode step ~30 次调用估算，切到 stable 约 +1.2 ms/step——这是必须显式接受的
   取舍，而不是白拿。
4. **唯一明显杠杆已定位**：把逐参数解包从 `to<Tensor>`（每个参数一个 shared_ptr）
   换成 handle 级解包。本次尝试 `to<AtenTensorHandle>` **段错误**（bare handle 的
   所有权语义与 `torch::stable::Tensor` 不同），已回退并记为 Phase 3 第一项待解问题；
   若解决，预计可回收 ~2us × 11 ≈ 20us，把 stable 直连压到 ~0.045–0.055，
   回到 Gate 附近。

## 8. 覆盖到 26/26 与三条门禁（2026-09-11 晚，910b + 950 双机实测）

§7.1/§7.3 之后的第二次刷新。本轮把最后一个算子接进来，并把"改了代码、
忘了配套动作"这一类问题变成门禁。

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 适配覆盖 | **26/26**（24 codegen + 2 手写） | `tools/stable_coverage.py --strict` 退出码 0 |
| 910b 全量 parity | **266 PASS / 0 FAIL**，250 个场景 + 11 条带原因 SKIP（见 §8.5） | `regression_stable_full.py`，`ALL PASS: full stable parity` |
| 950 专属 parity | **15 PASS / 0 FAIL**，4 个算子 | `regression_stable_a5.py`，`ALL PASS: Ascend950 stable parity` |
| Python API 契约 | 26 个算子 **0 漂移** | `tools/op_api_parity.py` |
| 产物 | 245 736 B、0 个 `_ZN2at/_ZN3c10`、38 个 `aoti_torch_*`、带构建戳 | 221 上 `nm -D` |
| 跨 torch 版本 | 2.9 头编出的 950 产物在 2.7.1 运行时跑通全部 A5 场景 | 241：py3.12+torch2.9 编译，py3.10+torch2.7.1 运行 |

本轮修掉的三个真问题（都不是"再跑一遍就好了"的偶发）：

1. **`npu_causal_conv1d` 的输出形状**：ctypes 的 `_infer_causal_conv1d_y` 依赖
   `run_mode`/`head_num`/`x.dim()`，spec 用一条 `alloc` 表达式原样搬运，门面按
   `at::empty`/`empty_like` 承接。prefill、`head_num` 重排、update、spec-decode、
   width3 五种形态与 ctypes 逐位一致，含 `conv_states` 原地写回。
   varlen 的 `query_start_loc` 形态在 A2 上被内核拒绝（aclnn 561002），
   **两条路径同样被拒**，因此记为 SKIP 而不是通过。
2. **入参的 `when` 条件被 stable 生成器忽略**：`fwd_prepare` 的 `a_log`/`dt_bias`
   只有在 `use_gate_in_kernel` 为真时才允许下发（ctypes 写成 `arg if flag else None`），
   stable 之前会把非空指针原样传下去，A5 报 161002。生成器现在按条件把描述符
   置空，并把条件变量提前声明——它可能排在引用它的参数之后。
3. **stack 下标**：把条件变量提到前面时，顺手丢掉了"stream 位于最后一个输入槽"
   的记账；编译照样通过，A5 首次调用就 segfault。现在 `generate_cpp` 在生成后断言
   读取下标恰为 `0..len(params)`、写回恰为 `0..outputs-1`，并已用负例验证能拦下。

另外，本轮把 **`.so` 与 Python glue 必须同源** 做成硬检查：`.inc` 的 md5 由
`build_stable.py` 编进库（导出 `fla_npu_thin_source_hash()`），`_stable.load()` 与
`_stable_generated._GENERATED_HASH` 比对，不一致直接报错并给出重编命令。
起因是改了 `.inc` 没重编 `.so`，三次实跑结果作废、多花了一轮排查。

Python 侧 API 契约的修法也记在这里：`tools/op_api_parity.py` 用 `ast` 把
`_aclnn_ctypes.py` 与 stable 后端逐参数比对，本轮抓出 10 个算子的漂移，其中 6 个
会直接抛 TypeError（位置参数默认值丢失、参数被漏掉、位置参数被改成关键字、
关键字改名、默认值语义从 False 变成 None）。修法不是逐个打补丁，而是让
`tools/sync_spec_python.py` 从 ctypes 反推 spec 的 `python` 块，并让生成器支持
"位置参数带默认值"与"必填关键字参数"。

### 8.1 覆盖记录入库、两条构建路径都验过（2026-09-11 收尾）

`tests/stable_scenarios.json` 把"这次跑了哪些场景"变成仓库里的一份事实：
910B3 记 **245 通过 + 2 条带原因的 SKIP**，950PR 记 **12 通过**。写用
`FLA_NPU_BASELINE_WRITE=1`，默认模式比对——**少一个场景就 FAIL**（丢 layout、
丢 flag 组合这类"绿着丢覆盖"的问题因此挡在门外）。重复跑一次确认基线自洽：
`baseline ok for Ascend910B3: 245 scenarios (0 new, 0 new skip)`。

另外两条构建路径都实测过，避免"改了默认把开关路径弄坏"：

| 构建 | 产物 | 安装后 |
| --- | --- | --- |
| 默认 | `...-910b.aarch64-py3-none-any.whl`，含 `libfla_npu_thin.so`、无 `_C_thin` | 无 `FLA_NPU_STABLE_LIB` / `ASCEND_CUSTOM_OPP_PATH` 也能跑全量 256 PASS |
| `FLA_NPU_BUILD_THIN=1` | `...-910b.aarch64-cp311-cp311-linux_aarch64.whl`，含 `_C_thin*.so` + `libfla_npu_thin.so` + torch pin | `FLA_NPU_THIN_ABI=pybind` → `_thin`；不设该变量 → 仍然 `_stable_generated` |

`test_wheel_environment.py` 也补了 4 个用例覆盖这对开关（默认必须是 ABI-free、
pybind 必须 opt-in、`FLA_NPU_BUILD_STABLE_ABI=0` 能出纯 ctypes wheel、
`build_wheel.py` 会丢掉残留的 `_C_thin*.so`），18 个用例全过。
`regression_mutation_contract.py`（version 计数 / grad 拒绝 / scratch state）在新的
默认链路上同样全过。

### 8.2 API 契约覆盖到 pybind，三条后端一致（2026-09-11 最后一项）

`FLA_NPU_THIN_ABI=pybind` 仍然可选，所以它的 Python 表面也是对外 API 的一部分。
`tools/op_api_parity.py` 改成**逐后端**比对（不再"谁先找到算子就只比谁"），于是把
pybind 的 9 处漂移也照出来了：位置默认值丢失（`causal_conv1d_bwd`、
`chunk_gated_delta_rule_fwd_prepare`、`chunk_kda_fwd`、`recurrent_kda`）、
参数漏掉（`bwd_dhu.transpose_state_layout`）、关键字改名
（`chunk_local_cumsum.chunk_indices` → ctypes 的 `chunk_indices_out`）、
多出 ctypes 没有的参数（`chunk_gated_delta_rule_fwd.a_log/dt_bias`）。

修完后的记录（51 = 26 算子 × 能提供它的后端）：

```
operators compared: 51    drifted: 0
SIGNATURES MATCH: every backend exposes the ctypes call shape
```

一处细节值得记：`bwd_dhu` 的 `transpose_state_layout` 在 pybind ABI 里本来就不存在
（pybind codegen 会跳过 cpp_only 参数），而 ctypes 是**收下但忽略**。所以 `_thin.py`
把它加进 Python 签名、但不转发给扩展——与 ctypes 行为一致，而不是硬塞进 ABI。

pybind 这一侧用 `tests/regression_thin_ops.py`（ctypes vs `_thin`）实测：
**26 个场景 ALL PASS / 246 PASS**，其中 6 个 conv1d 场景显式 SKIP（pybind 覆盖 25/26，
它本来就没有 `npu_causal_conv1d`）。

### 8.3 新增门禁：spec 的 aclnn 实参列表 vs 实现（2026-09-11 收尾）

本轮的教训来自 #390：它把 `aclnnCausalConv1d` 的 ABI 换了（4 个 `aclIntArray` 元数据槽
变成 `aclTensor`、`activationMode` 变成 `const char*`、多了 `nullBlockId`/`maxQueryLen`），
而我们的 spec 还写着旧表——**直到内核调用挂掉才发现**。既有的
`op_abi_validate.py` 能查这类偏差，但它需要 aclnn 头文件，而头文件只在配套 OPP 里；
我们手上恰好是旧 OPP，于是这个检查形同不存在。

`tools/op_abi_parity.py` 改为读**实现**：解析 `_aclnn_ctypes.py` 里每个算子
`_call_aclnn` 的实参列表（`lambda ctx: [...]`、本地 `build_args`、以及 conv1d 那种
"算子 → 共享 launch helper"的一层间接），把每个实参归类成
`tensor/int_array/int64/double/bool/char_ptr`，再与 spec 的 `args` 逐位比对。
静态解析不了的地方报 UNRESOLVED 而不是猜。

实测（同一份 spec、两份实现）：

```
当前分支（#390 之前）：specs checked: 26   mismatched: 0   ABI MATCH
main（#390 之后）：    specs checked: 26   mismatched: 1
    npu_causal_conv1d: aclnn arguments differ
        spec: tensor×4, int_array×4, int64×4, tensor                    (13)
        impl: tensor×8, int_array×4, char_ptr, int64×5, tensor          (19)
```

也就是说，这条门禁在**没有 OPP 头文件**的情况下，能提前把"spec 与实现的 ABI 脱节"
指出来，并且把两侧的实参表都打印出来——下次合 main 时它会先响，而不是等内核崩。
`tests/test_stable_gates.py` 现在 13 个用例：既查当前树为绿，也用一个被改过的实现
副本验证它确实会响。

### 8.4 公共 API 全量演练与回退计数（C4）

把驱动里的"钉住后端"换成"走公共 API"（`FLA_NPU_DISPATCH=public`），整套场景就覆盖到
后端选择 + mutation 契约 + launcher 三条路径，跑完断言 `FALLBACKS` 为空：

```
public dispatch: 24 operators, backends ['stable'], no fallback
ALL PASS: full stable parity          (当时 259 PASS，基线 246 + 3 条带原因 SKIP)
```

顺带把四种模式的行为钉住了：默认 `stable`；`FLA_NPU_THIN_TRACE=1` 逐算子打印
`[fla-npu] <op>: stable`；`FLA_NPU_THIN_ABI=ctypes` 显示 `ctypes` 但**不计**回退
（那是显式选择）；`FLA_NPU_THIN_VALIDATE=1` 显示 `ctypes` 并记一次回退（带原因）。
这样"某个算子悄悄退回 ctypes"这件事从"没人会发现"变成"跑一次就报"。

### 8.5 场景广度补强：把"声明了但没跑"找出来（2026-09-11 收尾）

`tools/coverage_gap_report.py` 按**算子**对齐"声明的合法域"与"跑过的场景"——
全局字符串匹配会骗人（别的算子跑过 BSND 会让 `chunk_fwd_o` 看起来也跑过），
所以场景名先按算子过滤，再逐轴比对。它找出的缺口逐条处理：

| 算子 | 结果 |
| --- | --- |
| `chunk_fwd_o` | 新增 **NTD** 通过；BSND / TND / `use_exp2` / `transpose_state_layout` → 内核 161001，两条后端一致 → SKIP |
| `causal_conv1d_bwd` | 新增 BSND / TND / NTD → 内核 561002（两条后端一致）→ SKIP；本 OPP 只有 BNSD 可用 |
| `chunk_local_cumsum` | 新增 `output_dtype=bfloat16` 通过；`head_first=False` → 161001 → SKIP |
| `recurrent_kda` | 新增 **TND**（含 in-place state 一致）通过 |
| `chunk_kda_bwd_intra` | 新增 **BSND** 通过 |
| `chunk_gated_delta_rule_fwd` | 场景标签补上 layout（`BNSD_B2_...`），覆盖记录才能读出来 |

改完的基线：910B3 **256 通过 + 16 条带原因 SKIP**，全量 **272 PASS**。
SKIP 全部带具体原因，而不是"跳过"。这一批里还多出两类值得记的：

* `chunk_fwd_h(save_new_value=False)`、`chunk_kda_bwd(disable_recompute=False)`、
  `chunk_kda_bwd(state_v_first=True)`——**参考实现用 Python 校验就拒了**
  （`save_new_value must be True` / `disable_recompute=... must be ...`），而 launcher
  把参数交给内核、由内核拒。两者都是报错、不崩，正好落在我们写的契约
  （"非法输入报错、不必同型"）上；`parity_or_domain_skip` 把这两种情形分开记录，
  避免把"内核之间的拒绝不一致"也当成正常。
* `chunk_bwd_dqkwg(use_exp2/transpose_state_layout)`——两条后端都是 **561000**。

两个新发现，都记进了文档而不是藏起来：

1. conv1d 家族在本 OPP 上只有最基础的那种形态可用（前向 varlen、反向三种 layout 全部
   561002）——这正好解释了 #390 为什么必须同时换 OPP。
2. `npu_solve_tri(layout="tnd")` 会让进程**静默崩溃**，ctypes 与 stable 都一样。
   崩溃没有可兼容的语义，所以三处（ctypes 参考、stable 的 `python.pre`、pybind 的
   `_thin.py`）都改成显式拒绝并说明原因；`scenario_solve_tri_guards` 钉住"两条后端
   都必须报错而不是崩"。

## 9. 并入 #390（rebase 到最新 main）后的适配

`origin/main` 的 tip 就是 #390 的 merge，所以这一步是"把分支推到 main 上并把它的
算子接进来"。它带来两件事：

1. **`aclnnCausalConv1d` 换 ABI**：4 个 `aclIntArray` 元数据槽变成 `aclTensor`、
   `activationMode` 变成 `const char*`、多了 `nullBlockId` / `maxQueryLen`；
2. **4 个新入口**：`causal_conv1d_fn`、`causal_conv1d_update`、
   `chunk_gdn_bwd_intra`、`chunk_kda_bwd_recompute`（旧的 `npu_causal_conv1d`
   变成弃用壳，三个 API 共用一个 ABI）。

### 9.1 conv1d 家族：Python 层共用，只把发射搬进 launcher

三个公开 API 的差别只在"怎么把参数拼成那一个 ABI"（校验、CPU 元数据数组、activation
字符串、update 的结果 copy 回 `x`），这层重复写一遍没有意义，也会随时间漂移。所以：

* launcher 侧把这**一个 ABI** 做成内部 op `_causal_conv1d_launch`（spec 标
  `"internal": true`，用 `"impl": "_launch_causal_conv1d"` 指明"谁拥有 aclnn 调用"，
  这样 ABI 门禁仍然会盯着它）；
* ctypes 的 `_launch_causal_conv1d` 在 launcher 已加载时把**发射**交给它，否则自己
  走 `_call_aclnn`；
* `_stable.py` 把三个公开函数**直接 re-export**（同一个函数对象），所以 Python 表面
  不可能漂移，`op_api_parity` 对它们是"同一对象"而不是"看起来一样"。

### 9.2 另外两个新算子

`npu_chunk_gdn_bwd_intra`、`npu_chunk_kda_bwd_recompute` 按常规 spec 生成适配器。
但**它们的 kernel 不在任何可用 OPP 里**（扫过 0908 下所有 `libcust_opapi.so`，
符号数为 0），所以只能交付代码、不能交付验证——这一点写在下面的环境注记里，
`coverage_gap_report.py` 也会把它们列成"没有场景"。

### 9.3 验证结果与环境注记（910B3 / 221）

| 项 | 结果 |
| --- | --- |
| 离线门禁 | coverage 30 算子 0 缺口、API 契约 55 对 0 漂移、**ABI 契约 28 个 0 不符**、spec 同步 0 |
| 合并后全量（完整 OPP） | **259 PASS**，基线 **250 通过 + 22 条带原因 SKIP** |
| 其中 conv1d 反向 | 通过（它的 ABI 没变） |
| conv1d 前向 8 个场景 | 记为"需要 post-#390 的 OPP"，并在 #390 的 OPP(`env390`) 上**全部通过**：prefill ×2、update、spec-decode、width3、gather-padding，含 in-place state 与 pad-slot 语义 |
| `FLA_NPU_CONV1D_ABI=old` | 把"本 OPP 是 #390 之前的 ABI"变成一条记录，而不是 ABI 未定义的调用（实测会直接杀进程） |

**环境注记**：手上没有哪一份 OPP 同时具备"新 conv1d ABI"和"其余 25 个 kernel"——
`env390` 是只为 causal_conv1d 编的单算子 OPP（`nm` 里只有一个
`aclnnCausalConv1d`）。所以前向场景在 `env390` 上验证、在完整 OPP 的那一轮里记为 SKIP；
代码是统一的，环境不是。

而且合并后 950 侧也重新跑过一遍（此前那一轮用的是合并前的代码）：

```
baseline written for Ascend950PR_9579: 38 passed, 6 skipped
ALL PASS: Ascend950 stable parity
```

比合并前的 34+5 多了一组：合并带进来的新场景（`chunk_fwd_h` 的 flag 变体等）在 A5 上也跑了。
6 条 SKIP 与 910b 上同源——`chunk_fwd_o` 的四个组合、`head_first=False`、以及
`chunk_fwd_h(save_new_value=False)` 这条"参考实现用 Python 校验就拒"的记录。

**241 上的一个环境教训**：跑这轮时 `torch.npu.set_device(0)` 直接挂住（进程停在
`locks_lock_inode_wait`），`npu-smi` 显示 NPU 0 的 Health 变成 `Warning`。用
`ASCEND_RT_VISIBLE_DEVICES=3` 换到健康卡后一切正常——以后遇到"还没进场景就卡住"，
先看卡的 Health，而不是怀疑代码。

### 9.4 host 侧：与 ctypes、与 pybind 两把尺子（`bench_stable_host.py --baseline`）

同一套场景输入，只换"基准是谁"，就能回答两个不同的问题。

**对 ctypes（旧发货路径）**：24 个算子全部更快，比值 0.19×–0.39×（快 2.6–5.3 倍）。

**对 pybind（ABI 绑定的那套 thin）**：23 个算子，**21 个持平或更快**，只有 3 个略慢，
最差 1.24×：

| 算子 | pybind | stable | 比值 |
| --- | --- | --- | --- |
| `chunk_kda_fwd`（最差） | 0.1637 | 0.2030 | 1.24 |
| `chunk_gated_delta_rule_fwd` | 0.1502 | 0.1725 | 1.15 |
| `recurrent_gated_delta_rule` | 0.1195 | 0.1239 | 1.04 |
| `chunk_kda_bwd` / `chunk_fwd_o` / `kda_gate_cumsum` / `chunk_local_cumsum` | — | — | ≈0.97–0.99 |
| `recurrent_kda` / `chunk_fwd_h` / `chunk_gated_delta_rule_fwd_h` | — | — | 0.92–0.95 |
| `prepare_wy_repr_bwd(_full/_da)` / `chunk_bwd_dv_local` | — | — | 0.81–0.88 |
| `scaled_dot_kkt` / `fast_gelu` / `chunk_bwd_dqkwg` / `fast_gelu_backward` | — | — | **0.54–0.72** |

（完整 23 行在 `tests/bench_stable_vs_pybind_910b.json`。）

这张表还带来一个实打实的优化：把生成 wrapper 从"构造 dict + `_call` 按 `_SIG` 重排"
改成**按 schema 顺序直接位置调用**之后，`recurrent_gated_delta_rule` 从 1.14 → 1.04、
`chunk_kda_bwd` 从 1.13 → 0.99、`chunk_fwd_h` 从 1.05 → 0.96，宽算子（19 个张量参数）
从 1.30 → 1.24。改完之后 API 契约与 ABI 门禁仍是 0 漂移。

剩下那两个 1.1–1.2× 的算子是"参数最多的那些"：dispatcher 每个张量参数约 2 µs 的装箱
成本是结构性下限（vllm-ascend 同路线，实测 0.073 ms 也是这个量级）。

### 8.6 扩展后的场景集在 pybind 后端也跑了一遍

新增的场景（chunk_fwd_o 各 layout、solve_tri 守卫、recurrent_kda TND、kda_bwd_intra
BSND、conv1d_bwd 各 layout、cumsum 的 dtype/head_first）在 pybind 后端同样跑：
**250 PASS / ALL PASS（26 个场景）**。差异只有两类，都是记录而不是静默：

| 记录 | 数量 | 原因 |
| --- | --- | --- |
| `the selected backend does not carry npu_causal_conv1d` | 6 | pybind 扩展本来就只有 25/26 个算子 |
| `pybind backend rejects what the reference accepts: 161001` | 1 | `chunk_local_cumsum(output_dtype="bfloat16")`：ctypes 与 stable 都 OK（输出 bf16），**pybind 扩展报 161001**（实测 `output_dtype="float32"` 正常、`"bfloat16"` 失败，且扩展只接受字符串） |

第二条是这次才发现的后端差异：以前没人跑过 `output_dtype` 变体。pybind 是 opt-in 的
A/B 通道且正在退役（D2 已把它移出默认构建），所以这里的选择是**把它如实记录**，
而不是为了它拖住主路径；`GAP_TOLERANT_BACKEND` 只在 pybind 驱动里打开，
Stable-ABI 驱动保持严格——同一情形在 stable 上仍然是硬失败。

### 8.7 Ascend950 侧也扩到同一批算子（2026-09-11 收尾）

950 侧原先只有 4 个 A5 专属场景。查过 A5 OPP 的符号后（`aclnnChunkFwdO`、
`aclnnSolveTri`、`aclnnChunkLocalCumsum`、`aclnnRecurrentKda` 都在；
`aclnnCausalConv1d` / `aclnnRecurrentGatedDeltaRule` / `aclnnChunkKdaBwdIntra` 不在），
把能跑的同一批场景也挂到 A5 驱动上：

```
baseline written for Ascend950PR_9579: 34 passed, 5 skipped
ALL PASS: Ascend950 stable parity
```

从 **12 通过** 变成 **34 通过 + 5 条带原因 SKIP**（SKIP 是 chunk_fwd_o 的四个组合与
`head_first=False`，A5 内核同样 161001 拒绝）。A5 侧现在覆盖到它 OPP 里有的 10 个
算子：`chunk_fwd_h`、`chunk_fwd_o`、`chunk_gated_delta_rule_fwd(_h)`、`fwd_prepare`、
`bwd_finalize`、`chunk_local_cumsum`、`chunk_scaled_dot_kkt`、`recurrent_kda`、
`solve_tri`。

**A5 OPP 里第 11 个算子 `recompute_w_u_fwd` 故意不在列表里**：它的 A5 kernel 对这些
输入**永远不返回**（只用 ctypes 单测 + 180s 超时复现，所以是内核而不是我们的层）。
挂死没法从 host 侧变成异常，因此排除并在此记录。

顺带修了一处环境问题：241 上的包副本还停留在没有 `tnd` 守卫的版本，`solve_tri_guards`
一跑就打到了裸内核（挂住而不是报错）——同步包文件并重编 A5 产物后正常。
