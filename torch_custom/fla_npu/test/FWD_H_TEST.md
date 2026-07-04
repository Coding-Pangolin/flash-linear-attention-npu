# fwd_h GVA CPU dual 测试指南

## 环境（另一台机器必配）

```bash
conda activate wnc

# CANN 9.0 示例（huangjunzhe 布局，vendor 无 set_env.bash）
export CANN_SET_ENV=/data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
export CANN_OPP_LIB=/data/huangjunzhe/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib

source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0
```

`cases.json`：`fla/ops/ascendc/gdn/cases.json`

## 推荐：随机输入 CPU dual（9 unsupported cases）

不依赖 example dump，与 recompute_wu / bwd_dhu 同风格。

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests

# smoke（3 个小 case）
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --smoke --no-viz

# 9 case 全量（CPU fp64 很慢，建议 --no-viz）
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz

# 先跑小 case，跳过大 case（T>=32768 / B=711 / T=262144）
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz \
  --cases gva_var_2,phase_1_fix_11

# 单 case
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 \
  ./run_fwd_h_cpu_dual_casesjson.sh --no-viz --cases gva_fix_3
```

输出目录：`fla/ops/ascendc/gdn/dual_benchmark_logs/fwd_h/cpu_dual_casesjson/<timestamp>/`

## cases.json 9 unsupported 矩阵

| Case | B | T | V | 规模 | 建议 |
|------|---|---|---|------|------|
| gva_var_2 | 1 | 16384 | 256 | 中 | 先跑 |
| phase_1_fix_11 | 16 | 16384 | 128 | 中 | 先跑 |
| gva_fix_3 | 711 | 196 | 128 | 大(B) | 后跑 |
| phase_1_fix_12 | 8 | 32768 | 128 | 大 | 后跑 |
| phase_1_var_5 | 1 | 32768 | 128 | 大 | 后跑 |
| gva_var_3 | 1 | 65536 | 256 | 大 | 后跑 |
| gva_var_5 | 1 | 65536 | 128 | 大 | 后跑 |
| phase_1_var_6 | 1 | 65536 | 128 | 大 | 后跑 |
| gva_var_6 | 1 | 262144 | 256 | 极大 | 最后 |

## 备选：example dump 路径

需 `flash_gated_delta_rule.py` 前置能 dump；`gva_fix_3` 等 4 case 因 Triton UB 无法 dump。

```bash
cd torch_custom/fla_npu/test
FWD_H_SUITE=unsupported FWD_H_CASE=smoke_varlen_t256_v128 bash run_fwd_h_gva_cases.sh
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `FWD_H_CPU_DUAL_RANDOM` | 1 = 随机输入（不走 example dump） |
| `FWD_H_CPU_DUAL_OUT` | 输出目录 |
| `FWD_H_SUITE=unsupported` | example dump 路径的 9 case |
| `FWD_H_CASE` | 逗号分隔 case 名 |
| `FWD_H_VIZ` | example dump 路径 viz 开关 |

## 核心文件

- `run_fwd_h_cpu_dual_casesjson.py` / `.sh` — 随机输入 CPU dual 主入口
- `test_npu_fwd_h_gva.py` — example dump + dual
- `test_fwd_h.py` — CPU 标杆 `forward_h_trans_cpu`（fp64/npu/fp32）
- `setup_cann_env.sh` — 可移植 CANN env
