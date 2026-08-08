# 开发者指南

本文档面向在 `flash-linear-attention-npu` 仓库内进行开发的开发者，按场景拆分为：单独编译单算子、增加新算子、测试单算子、端到端验证，以及如何确认 wheel 来自最新源码。

- [场景 1：单独编译单算子（run 包）](#场景-1单独编译单算子run-包)
- [场景 2：一键编包单算子](#场景-2一键编包单算子)
- [场景 3：增加一个新算子（目录结构 + torch_custom）](#场景-3增加一个新算子目录结构--torch_custom)
- [场景 4：测试单算子](#场景-4测试单算子)
- [场景 5：端到端 Example/ST 验证](#场景-5端到端-examplest-验证)
- [如何确认新构建的 wheel 来自最新源码](#如何确认新构建的-wheel-来自最新源码)

## 场景 1：单独编译单算子（run 包）

当已经安装了完整 wheel（方式 A），只需快速替换少量算子的 Ascend C 产物时使用。
`--ops=op1,op2,...` 只会生成指定算子的 run 包；run 包安装时会把当前 run 包里的
`packages/vendors/fla_npu_transformer` 合并覆盖到当前 Python 环境已安装的
`site-packages/fla_npu/opp/vendors/fla_npu_transformer`，从而更新 `aclnn`、tiling、kernel 和相关配置。

```sh
# 编译一个或多个算子 run 包，--soc 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=chunk_fwd_o

# 如果 Python wrapper 也有修改，再单独编译 Python runtime wheel
cd torch_custom/fla_npu
python3 setup.py bdist_wheel
```

## 场景 2：一键编包单算子

在仓库根目录用 `pip wheel` 全量构建 wheel，构建流程会清理上一轮 `build/`、`build_out/`
和 `output/` 中间产物，不依赖 Git diff 或旧 CMake 状态决定编译范围：

```sh
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
```

单算子定位请使用场景 1 的 `bash build.sh --pkg ... --ops=<op>` 方式构建 run 包。

## 场景 3：增加一个新算子（目录结构 + torch_custom）

### 3.1 目录结构

新增一个算子通常涉及以下目录：

| 目录 | 作用 |
| --- | --- |
| `fla/ops/ascendc/<模块>/<算子>/` | Ascend C 算子实现（host / kernel / tiling / proto） |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` | Python ctypes wrapper |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` | `_ASCENDC_OPS` 注册 |
| `torch_custom/fla_npu/test/test_npu_<op>.py` | 算子测试脚本 |

### 3.2 为已有 Ascend C 算子新增 Python 调用接口

核心链路：

1. 在 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 中按 `aclnn_xxx.h` 签名新增 `npu_xxx(...)` wrapper。
2. 在 `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` 的 `_ASCENDC_OPS` 注册算子名，注册后自动导出 `npu_xxx` 及去掉 `npu_` 前缀的短名。
3. 新增测试 `torch_custom/fla_npu/test/test_npu_<op>.py` 并接入 `test.sh`。
4. 重新构建 wheel / run 包并安装验证。

详细步骤（含示例骨架、特殊参数、正反向绑定、mutation 契约）见
[`torch_custom/fla_npu/README.md`](../torch_custom/fla_npu/README.md)。

### 3.3 算子调用方式

推荐通过 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 导入对应算子；具体入参可参考
`torch_custom/fla_npu/test` 下的对应算子测试脚本。例如：

```python
import torch
import fla_npu
from fla_npu.ops.ascendc import chunk_bwd_dv_local

dv = chunk_bwd_dv_local(...)
```

## 场景 4：测试单算子

使用者快速验证安装可参见根 [README](../README.md) Step 4 的冒烟测试；本场景面向开发调试，运行全量或单个算子的测试任务：

```sh
# 运行测试
cd torch_custom/fla_npu/test
bash test.sh --device 0                      # 全量测试
bash test.sh --device 0 --op causal_conv1d   # 单个 AscendC 测试任务
```

`--op` 当前仅覆盖 `test.sh` 已接入的 AscendC 测试任务，可选值：

- `prepare_wy_repr_bwd_full`
- `chunk_gated_delta_rule_bwd_dhu`
- `chunk_bwd_dv_local`
- `causal_conv1d`
- `chunk_local_cumsum`
- `chunk_scaled_dot_kkt`
- `prepare_wy_repr_bwd_da`
- `chunk_bwd_dqkwg`
- `gdn_fwd_o`
- `gdn_fwd_h`
- `recompute_w_u_fwd`

## 场景 5：端到端 Example/ST 验证

使用者快速验证可参见根 [README](../README.md) Step 4 的端到端示例；本场景面向开发调试与新增 CI 用例。完成安装后，可以一键运行 GDN 模块。该示例会组装 GDN 相关前向/反向算子，覆盖 AscendC 和 Triton 调用链：

```sh
python examples/flash_gated_delta_rule.py
```

NPU CI 的 Example/ST 用例由 [`ci/example_st_cases.json`](../ci/example_st_cases.json) 管理。
当前默认启用 `case1_current_default`，shape 与上面的直接运行默认值一致；后续 GVA、`Vdim=256`
等泛化场景可以在该文件中新增用例，显式填写 `B`、`T`、`chunk_size`、`query_head`、
`value_head`、`Kdim`、`Vdim` 等 shape 字段，以及 `gate_source`、`gate_function`、
`initial_state`、`output_final_state`、`qk_l2norm` 等行为字段。

当前端到端 Example/ST 已支持 `gate_source=g`；`gk` / `g+gk` 先作为用例 schema 预留，待
NPU fwd_h 路径支持后再启用。

## 如何确认新构建的 wheel 来自最新源码

重新构建后，需要确认实际安装的 wheel 确实来自最新修改的源码，避免因版本号相同被
`pip` 跳过（详见根 README Step 3）。推荐以下几种方式组合验证：

### 方式 1：核对 wheel 文件名与版本号

```sh
# 构建时记录产物文件名（构建日志会输出准确文件名，勿用通配符）
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
ls -la dist/

# 安装后核对实际加载的版本
python -m pip show flash-linear-attention-npu
python -c "from importlib.metadata import version; print(version('flash-linear-attention-npu'))"
```

构建产物文件名包含 SOC / ABI 本地版本标签（见 `scripts/fla_npu_artifacts.py`），
`pip show` / `importlib.metadata` 的版本应与 `dist/` 下新产物的版本一致。

### 方式 2：确认实际加载的 OPP 路径

`import fla_npu` 会在 Python 进程内定位并加载 wheel 内嵌 OPP，并把实际加载的
`libcust_opapi.so` 路径写入环境变量 `FLA_NPU_OP_API_LIB`（同时把 vendor 目录注入
`ASCEND_CUSTOM_OPP_PATH`）。可打印这些值，确认走的是新安装的 wheel 而非旧路径：

```sh
python - <<'EOF'
import fla_npu
import os

print("module file:   ", fla_npu.__file__)
print("op api lib:    ", os.environ.get("FLA_NPU_OP_API_LIB"))
print("custom opp:    ", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
EOF
```

`FLA_NPU_OP_API_LIB` 应指向新 wheel 内 `site-packages/fla_npu/opp/.../libcust_opapi.so`。
结合 Step 4 的 `python scripts/check_packaged_wheel_api.py`，可确认公开 API 与 OPP 布局为
新产物。

### 方式 3：修改后强制覆盖安装

本地 dev 版本（如 `26.7.0.dev0`）重新构建后版本号可能不变，不带 `--force-reinstall` 的
`pip install` 会认为"已是最新版本"而跳过，导致实际仍是旧代码。务必使用强制覆盖：

```sh
WHEEL_PATH="dist/<新产物的准确wheel文件名>.whl"
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
```

如需进一步确认安装的是新代码，可在源码修改处临时打印标记（如 `print("[DEBUG] new build")`
或查看 `__pycache__` 的时间戳），确认输出后移除。
