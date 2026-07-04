# GDN 三算子双标杆测试总览

> 覆盖 **recompute_wu**、**fwd_h**、**bwd_dhu** 的 GPU dual 与 CPU dual 精度验证。  
> 分支：`feat/recompute-wu-gpu-dump-dual`（相对 `main` 增量合入）

---

## 1. 各算子测试文档

| 算子 | 文档路径 | 说明 |
|------|----------|------|
| recompute_wu | [`chunk_gdn_fwd/recompute_wu_fwd/test/RECOMPUTE_WU_TEST.md`](chunk_gdn_fwd/recompute_wu_fwd/test/RECOMPUTE_WU_TEST.md) | GPU dump dual + CPU random dual |
| fwd_h | [`chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/FWD_H_TEST.md`](chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/FWD_H_TEST.md) | GPU dump dual + CPU random/dump dual |
| bwd_dhu | [`chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/BWD_DHU_TEST.md`](chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/BWD_DHU_TEST.md) | GPU dump dual + CPU random dual |

合入级验证流程另见：[`docs/GDN_OPERATOR_VERIFICATION_GUIDE.md`](../../../../docs/GDN_OPERATOR_VERIFICATION_GUIDE.md)

---

## 2. 在 main 基础上需要添加的文件

以下文件均在 `feat/recompute-wu-gpu-dump-dual` 分支；**直接 checkout 该分支** 或 **cherry-pick 下列路径** 即可跑通三算子双标杆。

### 2.1 共享层（三算子共用，必加）

```
fla/ops/ascendc/gdn/cases.json                    # 42 项用例矩阵
fla/ops/ascendc/gdn/gdn_case_utils.py             # 用例加载、随机输入、GPU dump 约束
fla/ops/ascendc/gdn/gdn_cpu_dual_casesjson.py     # CPU dual 共享逻辑 + 8 项 CPU-only 名单
fla/ops/ascendc/gdn/gpu_dump_loader.py            # 读取 GPU .pt dump
fla/ops/ascendc/gdn/gpu_dump_dual_utils.py        # ct.dual / ct.viz 封装
fla/ops/ascendc/gdn/gpu_dump_dual_runner.py       # GPU dual 批跑、报告合并
fla/ops/ascendc/gdn/gpu_dump_dual_log.sh          # GPU dual 日志 tee
fla/ops/ascendc/gdn/run_gdn_gpu_dump_dual_all.sh   # 三算子 GPU dual 串行入口
fla/ops/ascendc/gdn/run_gdn_cpu_dual_casesjson.sh # 三算子 CPU dual 串行入口
torch_custom/fla_npu/test/setup_cann_env.sh       # 可移植 CANN + vendor env
```

### 2.2 recompute_wu 专用

```
fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test/run_recompute_wu_gpu_dump_dual.sh
fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test/test_recompute_wu_gpu_dump_dual.py
fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test/run_recompute_wu_cpu_dual_casesjson.sh
fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test/run_recompute_wu_cpu_dual_casesjson.py
```

### 2.3 fwd_h 专用

```
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/run_fwd_h_gpu_dump_dual.sh
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/test_fwd_h_gpu_dump_dual.py
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/run_fwd_h_cpu_dual_casesjson.sh
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/run_fwd_h_cpu_dual_casesjson.py
```

**torch_custom 备选入口（example dump 路径，可选）：**

```
torch_custom/fla_npu/test/test_fwd_h.py
torch_custom/fla_npu/test/test_npu_fwd_h_gva.py
torch_custom/fla_npu/test/run_fwd_h_gva_cases.sh
```

### 2.4 bwd_dhu 专用

```
fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/run_bwd_dhu_gpu_dump_dual.sh
fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/test_bwd_dhu_gpu_dump_dual.py
fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/run_bwd_dhu_cpu_dual_casesjson.sh
fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/run_bwd_dhu_cpu_dual_casesjson.py
```

**torch_custom 备选入口（与 7/3 log 一致，可选）：**

```
torch_custom/fla_npu/test/test_bwd_dhu.py
torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py
torch_custom/fla_npu/test/run_bwd_dhu_gva_cases.sh
```

### 2.5 GPU dump 采集依赖（GPU dual 必做）

GPU dual 需要竞品侧 dump 数据，不在 NPU 仓内生成：

| 项目 | 说明 |
|------|------|
| GPU 仓 | https://github.com/Coding-Pangolin/flash-linear-attention ，分支 `feat/gdn-gpu-dump` |
| dump 脚本 | GPU 仓内 `GDN_DUMP_GUIDE.md` |
| NPU 读取 | 将 dump 目录 rsync 到 NPU 机，传给 `run_*_gpu_dump_dual.sh <DUMP_ROOT>` |

`examples/flash_gated_delta_rule.py` 在本分支有 dump 相关改动；若 main 上版本过旧，需同步该文件或仅在 GPU 仓采集。

---

## 3. 环境准备（NPU 机）

```bash
conda activate wnc

# 按本机路径设置（二选一示例）
# CANN 8.5 + vendor set_env.bash：
export CANN_SET_ENV=/data/zs/run/8.5/ascend-toolkit/set_env.sh
export VENDOR_SET_ENV=/data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash

# CANN 9.0（vendor 可能无 set_env.bash，只需 OPP lib）：
# export CANN_SET_ENV=/data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
# export CANN_OPP_LIB=/data/huangjunzhe/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib

source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0    # 改成空闲 NPU
pip install ct             # ct.dual / ct.viz
```

算子须已编译安装（`build.sh --pkg` + `.run` 安装 + `torch_custom/fla_npu` whl）。

---

## 4. 一键跑三算子

### 4.1 GPU dual（需 GPU dump 目录）

```bash
cd fla/ops/ascendc/gdn

# 三算子串行，报告在 <DUMP_ROOT>/gdn_gpu_dump_dual_out/
TEST_DEVICE_ID=0 ./run_gdn_gpu_dump_dual_all.sh /path/to/GPU_DUMP

# 单算子 / 单 case
./chunk_gdn_fwd/recompute_wu_fwd/test/run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1
./chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/run_fwd_h_gpu_dump_dual.sh /path/to/GPU_DUMP --case gva_var_1
./chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/run_bwd_dhu_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1
```

**GPU dual 覆盖范围：** `cases.json` 中 `chunk_size=64` 且不在 CPU-only 8 项名单的 case（约 30 项/算子，以各算子输出张量为准）。

### 4.2 CPU dual（随机输入，无需 GPU dump）

```bash
cd fla/ops/ascendc/gdn

# 三算子串行 smoke
TEST_DEVICE_ID=0 ./run_gdn_cpu_dual_casesjson.sh --smoke --no-viz

# 三算子全量 8 项 CPU-only batch
TEST_DEVICE_ID=0 ./run_gdn_cpu_dual_casesjson.sh --no-viz

# 单算子
TEST_DEVICE_ID=0 ./run_gdn_cpu_dual_casesjson.sh --op recompute_wu --smoke --no-viz
TEST_DEVICE_ID=0 ./run_gdn_cpu_dual_casesjson.sh --op fwd_h --no-viz
TEST_DEVICE_ID=0 ./run_gdn_cpu_dual_casesjson.sh --op bwd_dhu --cases gva_fix_3 --no-viz
```

**CPU-only 8 项**（`gdn_cpu_dual_casesjson.DEFAULT_GPU_UNSUPPORTED_CASES`）：

`gva_fix_3`, `gva_var_2`, `gva_var_3`, `gva_var_5`, `gva_var_6`, `phase_1_var_4`, `phase_1_var_5`, `phase_1_var_6`

---

## 5. 双标杆策略对照

| 路径 | 标杆 | 输入来源 | 通过标准 |
|------|------|----------|----------|
| **GPU dual** | `ct.dual(npu, fp64_gt, gpu_bench)` | GPU example dump `.pt` | 各输出张量 L1 PASS |
| **CPU dual** | `ct.dual(npu, fp64_gt, npu_bench)` | `cases.json` 随机生成 | 各输出张量 L1 PASS |

L1 阈值：`MARE_ratio≤5.0`, `MERE_ratio≤1.5`, `RMSE_ratio≤1.5`, `ERR_COUNT_ratio≤2.0`。

---

## 6. 日志与报告输出

| 路径 | 内容 |
|------|------|
| `dual_benchmark_logs/recompute_wu/cpu_dual_casesjson/<ts>/` | recompute_wu CPU dual |
| `dual_benchmark_logs/fwd_h/cpu_dual_casesjson/<ts>/` | fwd_h CPU dual |
| `dual_benchmark_logs/bwd_dhu/cpu_dual_casesjson/<ts>/` | bwd_dhu CPU dual |
| `<DUMP_ROOT>/gdn_gpu_dump_dual_out/<op>/` | GPU dual JSON 报告 + viz |
| `torch_custom/fla_npu/test/bwd_dhu_out/` | torch_custom bwd_dhu 输出（可选） |

---

## 7. 从 main 快速启用（checkout 示例）

```bash
git fetch coding-pangolin feat/recompute-wu-gpu-dump-dual
git checkout feat/recompute-wu-gpu-dump-dual
# 或：git cherry-pick <commit-range>  # 仅合入测试脚本时
```

合入 main 前建议：GPU dual 全量 PASS + CPU dual 8 项跑完 + 附 [`bwd_dhu_gva_test_report_0703.md`](../../../../torch_custom/fla_npu/test/bwd_dhu_gva_test_report_0703.md) 类精度报告。
