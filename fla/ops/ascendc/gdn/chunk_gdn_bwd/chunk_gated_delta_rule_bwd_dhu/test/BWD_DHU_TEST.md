# bwd_dhu 双标杆测试指南

算子：`chunk_gated_delta_rule_bwd_dhu`  
输出：`dh`, `dv2`（`h0`/`dht` 本轮为 None）

总览与共享文件清单：[`fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md`](../../../GDN_DUAL_TEST_GUIDE.md)

---

## 1. main 基础上需添加的本算子文件

在共享层（见总览 §2.1）之外，**bwd_dhu 还需以下 4 个文件**：

| 文件 | 用途 |
|------|------|
| `run_bwd_dhu_gpu_dump_dual.sh` | GPU dual 入口 |
| `test_bwd_dhu_gpu_dump_dual.py` | 读 GPU dump，`ct.dual(npu, fp64, gpu)` |
| `run_bwd_dhu_cpu_dual_casesjson.sh` | CPU dual 入口（fla 侧，日志进 dual_benchmark_logs） |
| `run_bwd_dhu_cpu_dual_casesjson.py` | 随机输入 CPU dual |

路径前缀：`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/`

**可选 torch_custom 入口**（与历史 aclnn log 一致）：

| 文件 | 用途 |
|------|------|
| `torch_custom/fla_npu/test/test_bwd_dhu.py` | CPU 标杆 fp64/npu/fp32 |
| `torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py` | 用例 + dual + viz |
| `torch_custom/fla_npu/test/run_bwd_dhu_gva_cases.sh` | torch_custom 主入口 |

---

## 2. 环境

```bash
conda activate wnc
export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
export VENDOR_SET_ENV=/path/to/fla_npu_transformer/bin/set_env.bash   # 若有
source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0
```

`cases.json`：`fla/ops/ascendc/gdn/cases.json`（`BWD_HU_CASES_JSON` 可覆盖）

---

## 3. GPU dual

### 3.1 命令

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test

TEST_DEVICE_ID=0 ./run_bwd_dhu_gpu_dump_dual.sh /path/to/GPU_DUMP
./run_bwd_dhu_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1
./run_bwd_dhu_gpu_dump_dual.sh /path/to/GPU_DUMP --phase prefix:gva_ --no-viz
```

dump 文件：`<DUMP_ROOT>/<case_name>/` 下 bwd_dhu 对应 `.pt`（见 GPU 仓 dump 指南）。

### 3.2 覆盖范围

- `cases.json` 中 `chunk_size=64` 且 **不在** CPU-only 8 项的 case（约 30 项）
- **`dh` 与 `dv2` 均须 PASS**

---

## 4. CPU dual

### 4.1 fla 侧（推荐，日志统一）

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test

# smoke
TEST_DEVICE_ID=0 ./run_bwd_dhu_cpu_dual_casesjson.sh --smoke --no-viz

# 8 项 CPU dual
TEST_DEVICE_ID=0 ./run_bwd_dhu_cpu_dual_casesjson.sh --no-viz

# 全量 42 项 NPU-only
TEST_DEVICE_ID=0 ./run_bwd_dhu_cpu_dual_casesjson.sh --all-cases --npu-only --no-viz

# 单 case（gva_fix_3：B=711，CPU fp64 约 30min~1h+）
TEST_DEVICE_ID=0 ./run_bwd_dhu_cpu_dual_casesjson.sh --cases gva_fix_3 --no-viz
```

输出：`fla/ops/ascendc/gdn/dual_benchmark_logs/bwd_dhu/cpu_dual_casesjson/<timestamp>/`

### 4.2 torch_custom 侧（备选）

```bash
cd torch_custom/fla_npu/test

BWD_HU_SUITE=unsupported TEST_DEVICE_ID=0 bash run_bwd_dhu_gva_cases.sh
BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 TEST_DEVICE_ID=0 bash run_bwd_dhu_gva_cases.sh
BWD_HU_VIZ=1 BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 bash run_bwd_dhu_gva_cases.sh
```

### 4.3 CPU-only 8 项

`gva_fix_3`, `gva_var_2`, `gva_var_3`, `gva_var_5`, `gva_var_6`, `phase_1_var_4`, `phase_1_var_5`, `phase_1_var_6`

定义于 `gdn_cpu_dual_casesjson.DEFAULT_GPU_UNSUPPORTED_CASES`。

### 4.4 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TEST_DEVICE_ID` | 2 | NPU 设备号 |
| `BWD_DHU_CPU_DUAL_OUT` | dual_benchmark_logs 下 | fla CPU dual 输出 |
| `BWD_HU_SUITE` | builtin | `unsupported` = 8 项 cases.json |
| `BWD_HU_CASE` | 空 | 逗号分隔 case 名 |
| `BWD_HU_VIZ` | 0 | 1 = ct.viz |
| `BWD_HU_SAVE_OUT` | 1 | 0 = 不写 outputs.pt |
| `BWD_HU_DUAL_LEVEL` | L1 | ct.dual 等级 |

---

## 5. 精度报告

- 用例矩阵与 Excel 映射：[`torch_custom/fla_npu/test/bwd_dhu_gva_test_report_0703.md`](../../../../../../torch_custom/fla_npu/test/bwd_dhu_gva_test_report_0703.md)
- 7/3 快照：`gva_fix_3` dv2 FAIL（dh PASS）；`gva_var_5` PASS

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `test_chunk_gated_delta_rule_bwd_dhu.py` | legacy 单测 |
| `../README.md` | 算子说明 |
| `run_gdn_gpu_dump_dual_all.sh` | 三算子 GPU dual |
| `run_gdn_cpu_dual_casesjson.sh --op bwd_dhu` | 三算子 CPU dual |
