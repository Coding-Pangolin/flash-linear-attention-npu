"""recurrent_kda 的 ATK executor。

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


OP_NAME = "recurrent_kda"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    calc_dtype = torch.float64 if high_precision else torch.bfloat16
    seed = int(spec.get("seed", 20260817))
    B, T, H, HV, K, V = (int(spec[x]) for x in ("B", "T", "H", "HV", "K", "V"))
    return {
        "q": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 1),
        "k": _randn((B, T, H, K), "bf16", calc_dtype, device, seed + 2),
        "v": _randn((B, T, HV, V), "bf16", calc_dtype, device, seed + 3),
        "g": _kda_gate((B, T, HV, K), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 4),
        "beta": _rand((B, T, HV), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 5, 0.1, 0.9),
        "initial_state": _zeros((B, HV, V, K), "fp32", torch.float64 if high_precision else torch.float32, device),
        "cu_seqlens": _int_tensor([i * T for i in range(B + 1)], device, torch.int64),
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
        "layout": str(spec.get("layout", "BSND")),
    }


def _recurrent_kda_ref(inputs):
    q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
    state = inputs["initial_state"].clone()
    calc = torch.float64 if q.dtype == torch.float64 else torch.float32
    B, T, H, _ = q.shape
    HV, V = v.shape[2], v.shape[3]
    out = torch.zeros((B, T, HV, V), dtype=calc, device=q.device)
    group = max(HV // H, 1)
    state = state.to(calc)
    for b in range(B):
        for t in range(T):
            for hv in range(HV):
                h = hv // group
                s = torch.exp(g[b, t, hv].to(calc)).unsqueeze(0) * state[b, hv]
                kt = k[b, t, h].to(calc)
                delta = beta[b, t, hv].to(calc) * (v[b, t, hv].to(calc) - torch.matmul(s, kt))
                s = s + torch.outer(delta, kt)
                out[b, t, hv] = torch.matmul(s, q[b, t, h].to(calc) * float(inputs["scale"]))
                state[b, hv] = s
    return out.to(v.dtype), state.to(inputs["initial_state"].dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _recurrent_kda_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.recurrent_kda(inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], inputs["initial_state"], cu_seqlens=inputs["cu_seqlens"], ssm_state_indices=None, layout=inputs["layout"], scale=inputs["scale"], output_final_state=True, inplace_final_state=False, state_v_first=True)


@register("executor_recurrent_kda")
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
