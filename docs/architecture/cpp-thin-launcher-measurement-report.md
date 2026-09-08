# fla_npu C++ 薄层：方案、实测与 ctypes 差异报告

> 关联 PR：#496；设计文档：[cpp-thin-launcher-design.md](./cpp-thin-launcher-design.md)
> 关联 issue：[#491](https://github.com/flashserve/flash-linear-attention-npu/issues/491)
> 日期：2026-09-08

## 1. 结论摘要

1. 薄层把算子 host enqueue 从"Python/ctypes 解释层"搬进"编译型 C++"，同
   kernel、同 OPP，输出/state 与 ctypes **逐位一致**。
2. recurrent 实测：ctypes public ~0.52-0.65 ms → 薄层 public（M4 后）
   ~0.10 ms，薄层 C++ 直连 ~0.043 ms；vllm custom public ~0.073 ms。
3. conv1d（PR #390 ABI 验证环境）：ctypes ~0.67-0.89 ms → 薄层 ~0.074 ms，
   parity 逐位一致。
4. 兼容性：有效输入路径与 ctypes 完全等价（签名/mutation/数值/stream 顺序）；
   非法输入只保证"会报错"，不保证异常类型与 ctypes 一致（有意为之，薄层不做
   完整 Python 前置校验）。
5. 薄层默认编译、默认启用（编译成功且算子已适配时），未适配/未编译自动回退
   ctypes；编译期不依赖 torch_npu。

## 2. 方案回顾

```text
fla_npu.ops.ascendc.npu_xxx(...)        # 公共 API（不变）
   └─ _thin.py（默认启用；FLA_NPU_THIN_LAUNCHER=0 关闭）
        └─ fla_npu._C_thin（C++，默认编译；FLA_NPU_BUILD_THIN=0 关闭编译）
             ├─ Runtime：dlopen 包内 libcust_opapi.so/libopapi.so，符号缓存
             ├─ TensorDesc：ND + storage 基址/offset/strides（与 ctypes 同语义）
             └─ 两段式 aclnn*GetWorkspaceSize + aclnn* launch
未适配算子 / 扩展缺失 → 自动回退 ctypes
```

关键决策（详见设计文档）：

- 不依赖 torch_npu 头/库（编译期），stream 由 Python 侧传入；
- workspace 仍每调用现分配（vllm custom 同款），per-stream 池留 M4 后续；
- 不复刻 vllm-ascend 的 NPUBridge/static_cast，descriptor 全部自建；
- 保留 ctypes fallback 与 mutation/autograd 契约；
- 每算子只需一份 ABI 适配（~50-100 行），按 profile 增量接入。

## 3. 实测方法与环境

环境：192.168.9.221（Ascend 910B3，CANN 9.1.0，fzy：Python 3.11 /
torch 2.9.0 / torch_npu 2.9.0.post2）。

方法：隔离子进程分别加载 FLA 与 vllm custom（避免同名 aclnn 符号冲突）；
`time.perf_counter()` 包住单次调用（不 synchronize）统计 host enqueue，
排序取 P50；NPU event 用 `torch.npu.Event` 统计。warmup 10-20 次，
正式 100-200 次。

## 4. 实测结果

### 4.1 RecurrentGatedDeltaRule（非连续 state、q=1、200 次）

shape：state `(blocks=1448, 16, 128, 128)`，stride gap=16384，offset=12288。

| 路径 | host P50 | event P50 | 说明 |
|---|---|---|---|
| ctypes public（原实现） | ~0.52-0.65 ms | ~0.14-0.34 ms | batch 32-128；共享机波动 |
| 薄层 C++ 直连 | ~0.043-0.053 ms | 同 kernel | 不含 Python wrapper |
| 薄层 public（M4 前） | ~0.165-0.17 ms | 同 kernel | wrapper：current_stream + bind |
| 薄层 public（M4 后） | ~0.098-0.108 ms | 同 kernel | stream 缓存 + mutation 快路径 |
| vllm custom public | ~0.073 ms | ~0.14-0.28 ms | 对照 |

结论：host 从 ctypes 的 ~6-7x custom 降到 M4 后的 ~1.4-1.5x；C++ 执行体
本身已快于 custom。event（device）两侧基本一致，未复现 issue 中的 +22-31%
device 差距（隔离单算子场景）。

### 4.2 CausalConv1d（PR #390 ABI 验证环境，update 形态）

shape：x `[8, 4096]` bf16，weight `[4, 4096]`，conv_state `(blocks, 6, 4096)`
非连续（gap=565248），activation=silu，run_mode=1，预分配 out。

| 路径 | host P50 | parity |
|---|---|---|
| FLA ctypes（#390 wheel） | ~0.67-0.89 ms | 基准 |
| 薄层（#390 ABI，直接写 out） | ~0.074 ms | out/state max_abs = 0 |

说明：#390 未合入 upstream，conv1d 薄层当前只在验证环境按 #390 ABI 跑通；
正式落地等 #390 合入后重写 `csrc_thin` 的 conv1d 适配。

### 4.3 正确性 / 回归

- recurrent：ctypes vs thin 在 q=1/q=4、连续/非连续 state 下 out/state
  **逐位一致**（max_abs = 0）。
- mutation：state `_version` 递增、requires_grad 拒绝（与 ctypes 同契约）。
- determinism / stream 顺序：重复调用 bit 一致；后续依赖 kernel 顺序正确。
- 自动化测试：`test_thin_launcher.py` + `test_thin_customer_compat.py`
  （7 passed + 3 subtests）。

## 5. 与 ctypes 的差异对照

| 维度 | 原 ctypes | 薄层（thin） |
|---|---|---|
| host 执行 | Python 校验 + ctypes 对象管理 + 两段式调用 | C++ descriptor 建销 + 两段式调用 |
| 数值/语义（合法输入） | 基准 | 与 ctypes 逐位一致 |
| 每调用 Python 校验 | 完整 dtype/device/shape 校验 | 少量校验（性能路径） |
| 非法输入报错 | 多为 `TypeError`/`RuntimeError`（Python 校验产生） | 常为 aclnn `RuntimeError`；类型不保证一致 |
| stream | 每调用 `torch.npu.current_stream()` | 首次获取 + 跟踪 `set_stream` 缓存 |
| workspace | 每调用 `torch.empty` | 每调用 C++ `at::empty`（暂同策略） |
| 编译依赖 | 无（纯 Python） | 编译期仅 torch/CANN；不依赖 torch_npu |
| 默认行为 | 默认 | 默认编译、默认启用（可 `=0` 关闭），缺失自动回退 |
| in-place/autograd | 经 `_wrap_mutable_direct_op` | 同（mutation wrapper 增加位置参数快路径） |
| 新增算子成本 | 1 个 Python wrapper（+可选 argtypes） | C++ 适配 + pybind + `_thin.py` + 白名单 |
| 适用范围 | 全部算子 | 白名单内算子；其余回退 ctypes |

### 已知且有意的差异

1. 非法输入报错类型不保证一致（薄层为速度跳过 Python 前置校验，只保证会报错）。
2. 白名单外算子不享受薄层加速（仍 ctypes）。
3. conv1d #390 update 形态在 #390 合入前不在 upstream 分支启用。

## 6. 变更文件清单

- `torch_custom/fla_npu/csrc_thin/`：runtime / tensor_desc / ops_recurrent_gdn /
  ops_causal_conv1d（占位）/ pybind
- `torch_custom/fla_npu/setup.py`：`_C_thin` 默认编译分支
- `fla_npu/ops/ascendc/_thin.py`：薄层 Python 适配（含 stream 缓存）
- `fla_npu/ops/ascendc/__init__.py`：默认启用 + 白名单 + mutation 快路径
- `torch_custom/fla_npu/test/test_thin_launcher.py`、
  `test_thin_customer_compat.py`：dispatch/parity/兼容性测试
- `docs/architecture/cpp-thin-launcher-design.md`：设计文档

## 7. 后续计划

- 剩余 public wrapper 差距（~0.03 ms）：pybind 位置参数直传等（可选 M4.2）；
- conv1d：等 #390 合入后按新 ABI 重写并启用；
- device/event 差距：在服务级 profiling 场景复核是否仍存在；
- 按 profile 增量接入其他高频算子。
