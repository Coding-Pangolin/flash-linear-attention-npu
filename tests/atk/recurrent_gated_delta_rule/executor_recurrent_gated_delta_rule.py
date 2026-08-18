"""recurrent_gated_delta_rule 的 ATK executor。

输入生成、CPU 标杆、run_cpu、run_npu 和 FunctionApi 都放在本算子目录中。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _RCP_LN2,
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _int_tensor,
    _kda_gate,
    _marker_device,
    _num_chunks,
    _orig_dtype,
    _rand,
    _randn,
    _zeros,
)


OP_NAME = "recurrent_gated_delta_rule"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    calc_dtype = torch.float64 if high_precision else torch.bfloat16
    seed = int(spec.get("seed", 20260817))
    T, HK, HV, K, V = (int(spec[x]) for x in ("T", "HK", "HV", "K", "V"))
    block_num = int(spec.get("block_num", 1))
    return {
        "query": _randn((T, HK, K), "bf16", calc_dtype, device, seed + 1),
        "key": _randn((T, HK, K), "bf16", calc_dtype, device, seed + 2),
        "value": _randn((T, HV, V), "bf16", calc_dtype, device, seed + 3),
        "state": _zeros((block_num, HV, V, K), "bf16", calc_dtype, device),
        "beta": _rand((T, HV), "bf16", calc_dtype, device, seed + 4, 0.1, 0.9),
        "g": _randn((T, HV), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 5, scale=0.01),
        "actual_seq_lengths": _int_tensor([0, T], device, torch.int32),
        "ssm_state_indices": _int_tensor([0 for _ in range(T)], device, torch.int32),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


def _recurrent_gated_delta_rule_ref(inputs):
    q, k, v = inputs["query"], inputs["key"], inputs["value"]
    state = inputs["state"].clone()
    beta, g = inputs["beta"], inputs["g"]
    indices = inputs["ssm_state_indices"].detach().cpu().tolist()
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    T, HK, _ = q.shape
    HV, V = v.shape[1], v.shape[2]
    out = torch.zeros((T, HV, V), dtype=calc, device=q.device)
    group = max(HV // HK, 1)
    state = state.to(calc)
    for t in range(T):
        slot = int(indices[t])
        for hv in range(HV):
            hk = hv // group
            s = torch.exp(g[t, hv].to(calc)) * state[slot, hv]
            kt = k[t, hk].to(calc)
            delta = beta[t, hv].to(calc) * (v[t, hv].to(calc) - torch.matmul(s, kt))
            s = s + torch.outer(delta, kt)
            out[t, hv] = torch.matmul(s, q[t, hk].to(calc) * float(inputs["scale"]))
            state[slot, hv] = s
    return out.to(v.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _recurrent_gated_delta_rule_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.recurrent_gated_delta_rule(inputs["query"], inputs["key"], inputs["value"], inputs["state"], beta=inputs["beta"], scale=inputs["scale"], actual_seq_lengths=inputs["actual_seq_lengths"], ssm_state_indices=inputs["ssm_state_indices"], g=inputs["g"], gk=None)


@register("executor_recurrent_gated_delta_rule")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device == "cpu" and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return _finite_tuple(outputs)
