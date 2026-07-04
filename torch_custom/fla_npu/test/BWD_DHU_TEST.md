# bwd_dhu GVA CPU dual 测试指南

## 环境（另一台机器必配）

```bash
conda activate wnc

# 按本机 CANN 路径设置（不要用 tracy/fazhenyao 组合，vendor 可能缺算子）
export CANN_SET_ENV=/data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
export VENDOR_SET_ENV=/data/huangjunzhe/Ascend/ascend-toolkit/opp/vendors/fla_npu_transformer/bin/set_env.bash
export CANN_OPP_LIB=/data/huangjunzhe/Ascend/ascend-toolkit/opp/vendors/fla_npu_transformer/op_api/lib

source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0   # 改成空闲 NPU
```

`cases.json` 路径：`fla/ops/ascendc/gdn/cases.json`（可用 `BWD_HU_CASES_JSON` 覆盖）。

## 推荐入口

### 1. torch_custom 路径（与 7/3 NPU4 log 一致）

```bash
cd torch_custom/fla_npu/test

# cases.json 9 项 unsupported batch
BWD_HU_SUITE=unsupported bash run_bwd_dhu_gva_cases.sh

# 单 case：gva_fix_3（B=711, V=128, cs=128，CPU fp64 很慢，约 30min~1h+）
BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 bash run_bwd_dhu_gva_cases.sh

# 加 ct.viz（更慢，大图）
BWD_HU_VIZ=1 BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 bash run_bwd_dhu_gva_cases.sh

# 不落盘 outputs.pt
BWD_HU_SAVE_OUT=0 BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 bash run_bwd_dhu_gva_cases.sh
```

### 2. gdn 侧 CPU dual runner（同逻辑，日志进 dual_benchmark_logs）

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test
source ../../../../../../torch_custom/fla_npu/test/setup_cann_env.sh

./run_bwd_dhu_cpu_dual_casesjson.sh --smoke
./run_bwd_dhu_cpu_dual_casesjson.sh --cases gva_fix_3
./run_bwd_dhu_cpu_dual_casesjson.sh   # 默认 9 unsupported cases
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TEST_DEVICE_ID` | 5 | NPU 设备号 |
| `BWD_HU_SUITE` | builtin | `unsupported` = cases.json 9 项 |
| `BWD_HU_CASE` | 空 | 逗号分隔单/多 case |
| `BWD_HU_VIZ` | 0 | 1 = ct.viz |
| `BWD_HU_VIZ_SAMPLE_COUNT` | 200000 | viz 采样点数 |
| `BWD_HU_SAVE_OUT` | 1 | 0 = 不写 outputs.pt |
| `BWD_HU_OUT_DIR` | `bwd_dhu_out/` | 输出目录 |
| `BWD_HU_DUAL_LEVEL` | L1 | ct.dual 等级 |

## 7/3 已知结果（NPU4, CANN 8.5 zs vendor）

9 case batch：**5 PASS / 3 SKIP / 1 FAIL**

| Case | 结果 | 说明 |
|------|------|------|
| gva_fix_3 | **FAIL** | dh PASS，dv2 FAIL（small_err_ratio ~10x，非 NaN） |
| gva_var_5 | PASS | |
| phase_1_fix_11/12, phase_1_var_5/6 | PASS | |
| gva_var_2/3/6 | SKIP | 与内置 GVA 矩阵等价已测（v5 PASS） |

## 核心文件

- `test_npu_bwd_dhu_gva.py` — 用例 + dual + viz
- `test_bwd_dhu.py` — CPU 标杆（fp64 / npu / fp32）
- `run_bwd_dhu_gva_cases.sh` — 主入口
- `setup_cann_env.sh` — 可移植 CANN env
- `fla/ops/ascendc/gdn/cases.json` — case 定义
