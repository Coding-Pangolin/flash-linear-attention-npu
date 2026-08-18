# ChunkGatedDeltaRuleFwdH ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_h` 的 ATK 单算子工程，包含 `executor_chunk_gated_delta_rule_fwd_h.py`、`gen_chunk_gated_delta_rule_fwd_h.py`、`chunk_gated_delta_rule_fwd_h.yaml`、`atk_chunk_gated_delta_rule_fwd_h.json`。

## 输入约束

- `k/w` 必须为 `[B,HK,T,K]`，`u/v_new` 必须为 `[B,HV,T,V]`。
- `g` 与 `gk` 至少提供一个：`g=[B,HV,T]`，`gk=[B,HV,T,K]`。
- `h` 输出为 `[B,HV,Nc,K,V]`，`state_v_first=true` 时末两维为 `[V,K]`；`initial_state/final_state` 同样受 `state_v_first` 解释。
- `k/w/u` 的 `B`、`T` 必须一致；`u` 的 `HV` 必须大于等于 `HK` 且 `HV % HK == 0`。
- `k/w/u/h/v_new` 支持 `BFLOAT16/FLOAT16`；gate 支持 `FLOAT/FLOAT16/BFLOAT16`，state 支持 `FLOAT/BFLOAT16/FLOAT16`。
- `V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`；变长模式支持 `cu_seqlens/chunk_indices` 成对传入。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_fwd_h.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_gated_delta_rule_fwd_h.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 64, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_fwd_h", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "HK": 1, "HV": 1, "T": 64, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_fwd_h", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
