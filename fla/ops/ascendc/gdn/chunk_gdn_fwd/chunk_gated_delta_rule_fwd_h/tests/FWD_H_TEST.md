# fwd_h 双标杆测试指南

算子：`chunk_gated_delta_rule_fwd_h`  
输出：`h`, `v_new`

总览与共享文件清单：[`fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md`](../../../GDN_DUAL_TEST_GUIDE.md)

---

## 1. main 基础上需添加的本算子文件

在共享层（见总览 §2.1）之外，**fwd_h 还需以下 4 个文件**：

| 文件 | 用途 |
|------|------|
| `run_fwd_h_gpu_dump_dual.sh` | GPU dual 入口 |
| `test_fwd_h_gpu_dump_dual.py` | 读 GPU dump，`ct.dual(npu, fp64, gpu)` |
| `run_fwd_h_cpu_dual_casesjson.sh` | CPU dual 入口 |
| `run_fwd_h_cpu_dual_casesjson.py` | 随机输入 CPU dual |

路径前缀：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/`

**可选 torch_custom 入口：**

| 文件 | 用途 |
|------|------|
| `torch_custom/fla_npu/test/test_fwd_h.py` | CPU 标杆 |
| `torch_custom/fla_npu/test/test_npu_fwd_h_gva.py` | example dump + dual |
| `torch_custom/fla_npu/test/run_fwd_h_gva_cases.sh` | torch_custom 主入口 |

---

## 2. 环境

```bash
conda activate wnc
export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
export CANN_OPP_LIB=/path/to/fla_npu_transformer/op_api/lib   # CANN 9.0 常用
source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0
```

`cases.json`：`fla/ops/ascendc/gdn/cases.json`

---

## 3. GPU dual

### 3.1 命令

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests

TEST_DEVICE_ID=0 ./run_fwd_h_gpu_dump_dual.sh /path/to/GPU_DUMP
./run_fwd_h_gpu_dump_dual.sh /path/to/GPU_DUMP --case gva_var_1
./run_fwd_h_gpu_dump_dual.sh /path/to/GPU_DUMP --phase prefix:phase_1_ --no-viz
```

### 3.2 覆盖范围

- `cases.json` 中 `chunk_size=64` 且不在 CPU-only 8 项的 case
- 输出 `h`、`v_new` 均须 L1 PASS

---

## 4. CPU dual

fwd_h 有两种 CPU dual 模式：

| 模式 | 触发 | 输入 | 适用 |
|------|------|------|------|
| **随机输入（推荐）** | `FWD_H_CPU_DUAL_RANDOM=1` | `cases.json` 随机生成 | 8 项 CPU-only，无需 dump |
| **example dump** | 默认（不设 RANDOM） | `flash_gated_delta_rule.py` dump | 需前置 Triton 能 dump 的 case |

### 4.1 随机输入（推荐）

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests

# smoke
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --smoke --no-viz

# 8 项全量
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz

# 先跑小 case
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz --cases gva_var_2,phase_1_var_4

# 单 case
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz --cases gva_fix_3
```

输出：`fla/ops/ascendc/gdn/dual_benchmark_logs/fwd_h/cpu_dual_casesjson/<timestamp>/`

### 4.2 example dump 路径

```bash
cd torch_custom/fla_npu/test
FWD_H_SUITE=unsupported FWD_H_CASE=smoke_varlen_t256_v128 bash run_fwd_h_gva_cases.sh
```

`gva_fix_3` 等部分 case 因 Triton UB 无法 dump，须走随机输入 CPU dual。

### 4.3 CPU-only 8 项与规模建议

| Case | B | T | V | 建议顺序 |
|------|---|---|---|----------|
| gva_var_2 | 1 | 16384 | 256 | 先跑 |
| phase_1_var_4 | 1 | 8192 | 128 | 先跑 |
| gva_fix_3 | 711 | 196 | 128 | 后跑 |
| phase_1_var_5 | 1 | 32768 | 128 | 后跑 |
| gva_var_3 | 1 | 65536 | 256 | 后跑 |
| gva_var_5 | 1 | 65536 | 128 | 后跑 |
| phase_1_var_6 | 1 | 65536 | 128 | 后跑 |
| gva_var_6 | 1 | 262144 | 256 | 最后 |

### 4.4 环境变量

| 变量 | 说明 |
|------|------|
| `FWD_H_CPU_DUAL_RANDOM` | 1 = 随机输入（推荐） |
| `FWD_H_CPU_DUAL_OUT` | 输出目录 |
| `FWD_H_SUITE=unsupported` | torch_custom 8 项 |
| `FWD_H_CASE` | 逗号分隔 case 名 |
| `FWD_H_VIZ` | example dump 路径 viz |
| `GDN_FWD_H_DUMP_DIR` | example dump 缓存目录 |

---

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `pta/test_fwd_h.py` | legacy PTA 路径 |
| `../README.md` | 算子说明 |
| `run_gdn_gpu_dump_dual_all.sh` | 三算子 GPU dual |
| `run_gdn_cpu_dual_casesjson.sh --op fwd_h` | 三算子 CPU dual |
