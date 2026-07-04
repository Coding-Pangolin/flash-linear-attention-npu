# bwd_dhu 测试指南（torch_custom 入口）

> **完整文档（含 GPU dual + 文件清单）：**  
> [`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/BWD_DHU_TEST.md`](../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/BWD_DHU_TEST.md)  
> **三算子总览：** [`fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md`](../../../fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md)

本目录提供 **torch_custom / aclnn** 路径的 CPU dual 入口，与 fla 侧 `run_bwd_dhu_cpu_dual_casesjson.sh` 逻辑一致。

## 快速开始

```bash
conda activate wnc
export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0

cd torch_custom/fla_npu/test
BWD_HU_SUITE=unsupported bash run_bwd_dhu_gva_cases.sh
BWD_HU_SUITE=unsupported BWD_HU_CASE=gva_fix_3 bash run_bwd_dhu_gva_cases.sh
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `BWD_HU_SUITE=unsupported` | cases.json 8 项 CPU-only |
| `BWD_HU_CASE` | 逗号分隔 case |
| `BWD_HU_VIZ` | 1 = ct.viz |
| `BWD_HU_SAVE_OUT` | 0 = 不写 outputs.pt |

精度报告：[`bwd_dhu_gva_test_report_0703.md`](bwd_dhu_gva_test_report_0703.md)
