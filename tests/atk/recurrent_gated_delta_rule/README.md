# RecurrentGatedDeltaRule ATK 工程

本目录提供 `recurrent_gated_delta_rule` 的 ATK 单算子工程，包含 `executor_recurrent_gated_delta_rule.py`、`gen_recurrent_gated_delta_rule.py`、`recurrent_gated_delta_rule.yaml`、`atk_recurrent_gated_delta_rule.json`。

## 输入约束

- `query/key` 必须为 `(T,Nk,Dk)`，`value/out` 必须为 `(T,Nv,Dv)`。
- `beta` 必须为 `(T,Nv)`；`stateRef` 为原地输入输出，shape 为 `(BlockNum,Nv,Dv,Dk)`。
- `actualSeqLengths` 必须为 `(B+1,)` 的 `INT32` 张量，首元素是不参与计算的无效长度，其余 `B` 个元素之和等于 `T`。
- `ssmStateIndices` 必须为 `(T,)` 的 `INT32` 张量，取值范围为 `[0, BlockNum)`。
- `query/key/value/beta/stateRef/out` 当前仅支持 `BFLOAT16`。
- 每条序列有效 token 数需要 `<= 8`；`g` 如提供为 `(T,Nv)` FP32，`gk` 当前未支持，必须传 `None`。
- `numAcceptedTokens` 如提供，shape 为 `(B,)`，每项不超过对应序列有效 token 数；`scale` 建议按 `1 / sqrt(Dk)` 设置。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta/golden.py; fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_gated_delta_rule.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "T": 2, "HK": 1, "HV": 1, "K": 128, "V": 128, "block_num": 1, "op": "recurrent_gated_delta_rule", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
