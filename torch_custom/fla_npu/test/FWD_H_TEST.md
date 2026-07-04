# fwd_h 测试指南（torch_custom 入口）

> **完整文档（含 GPU dual + 文件清单）：**  
> [`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/FWD_H_TEST.md`](../../../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/FWD_H_TEST.md)  
> **三算子总览：** [`fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md`](../../../fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md)

## 推荐：fla 侧随机输入 CPU dual

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests
FWD_H_CPU_DUAL_RANDOM=1 TEST_DEVICE_ID=0 ./run_fwd_h_cpu_dual_casesjson.sh --smoke --no-viz
```

## torch_custom example dump 路径

```bash
conda activate wnc
source torch_custom/fla_npu/test/setup_cann_env.sh
cd torch_custom/fla_npu/test
FWD_H_SUITE=unsupported FWD_H_CASE=smoke_varlen_t256_v128 bash run_fwd_h_gva_cases.sh
```

详见 fla 侧 [`FWD_H_TEST.md`](../../../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/FWD_H_TEST.md) 第 4 节。
