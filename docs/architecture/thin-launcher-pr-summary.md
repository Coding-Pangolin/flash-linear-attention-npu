# Thin Launcher（PR #496）方案与验证汇总

## 1. 目标与范围

把 `fla_npu.ops.ascendc.*` 的 Python/ctypes 两段式 `aclnn` 热路径替换为
C++ thin launcher：编译期只依赖 torch（C++ extension），CANN aclnn/opapi
符号运行时 `dlopen`（custom `libcust_opapi.so` + `libopapi.so`），消除
热路径上 Python/ctypes descriptor 建销与逐参转换。

## 2. 设计决策（与早期讨论一致）

| 主题 | 决策 |
| --- | --- |
| ctypes 源码 | 保留，作为回退/对照（`FLA_NPU_THIN_LAUNCHER=0` 全量回退） |
| 合法输入域 | thin 只在合法输入启用；非法/未适配组合自动回退 ctypes（不保证同型报错） |
| 接入方式 | JSON-only：加一个 `op_specs/*.json` → `setup.py` thin 构建前自动 codegen（cpp/pybind/_thin wrapper/动态白名单） |
| 动态白名单 | 由 `_thin` 模块函数推导（`_get_thin_op`），无硬编码清单 |
| 依赖 | 编译期 torch-only；运行期仍依赖 CANN + torch_npu（tensor/stream 语义） |
| 发布 | wheel 含 `cpXXX-linux_{x86_64,aarch64}` 扩展，按 Python 版本 × 平台矩阵产包 |

## 3. Codegen 能力（已落地并有算子验证）

- fragment 聚合幂等（单/多输出返回类型均可去重）；
- 输出分配：source/dtype/shape/`expr`/`alloc`（原始 C++ 表达式）；
- 条件输出 `when`/`when_py`/`return_when`（C++ null descriptor 与 Python None 映射解耦）；
- `return_order`（aclnn 输出序 → Python 返回序）与 `return_suffix`（尾部追加输入）；
- spec `helpers`（C++ 函数，如 chunk count）、`python.pre`（Python 预置/委托）、
  `ignored`（API 兼容但不进 aclnn 的参数）、标量 None 默认解析、varlen chunk 派生；
- `enabled: false`：暂不启用（如 solve_tri）时自动 codegen 跳过、保持 ctypes。

## 4. 实机验证（221 / 910B3，CANN 9.1.0，torch 2.9 + torch_npu 2.9.0.post2）

20 个算子 parity 0.0（ctypes vs thin），host P50 明细见
[thin-migration-inventory.md](thin-migration-inventory.md)。安装态数值回归
[../../tests/regression_thin_ops.py](../../tests/regression_thin_ops.py)
覆盖 16 场景/19 组 parity，安装态 smoke
（`torch_custom/fla_npu/test/test_wheel_install_smoke.py`）覆盖 22 个 thin
算子 dispatch 门控，均全绿。

## 5. 边界与已知项

- 部分算子合法域受 wrapper 主机语义限制（多发射分段/补齐/复制），thin 覆盖其
  单发射子域并回退其余（chunk_kda_bwd/bwd_intra、recurrent 等）；
- conv1d legacy 保持 ctypes 至上游 #390；`causal_conv1d_update` 适配在验证分支；
- `fwd_prepare`/`bwd_finalize` 为 Ascend950-only；recurrent_kda 需本地测试资产；
- `solve_tri` thin parity 未修（enabled=false）。

## 6. 建议合入前检查

1. `python scripts/build_wheel.py --wheel-dir dist`（910b/950 各编一次）；
2. 安装后跑 regression_thin_ops + test_wheel_install_smoke；
3. 950 环境补 fwd_prepare/bwd_finalize/recurrent_kda 用例后更新本清单；
4. #390 合入后把 conv1d_update 适配并入本分支。
