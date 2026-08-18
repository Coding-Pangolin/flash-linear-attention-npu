"""chunk_gated_delta_rule_bwd_dhu 的 ATK executor。

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


OP_NAME = "chunk_gated_delta_rule_bwd_dhu"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "q": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 2),
        "w": _zeros((B, HV, T, K), dtype_name, calc_dtype, device),
        "do": _zeros((B, HV, T, V), dtype_name, calc_dtype, device),
        "dv": _zeros((B, HV, T, V), dtype_name, calc_dtype, device),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 3),
        "chunk_size": chunk_size,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(K))),
    }


def _zero_h_ref(inputs):
    if "u" in inputs:
        B, _, T, K = inputs["k"].shape
        HV, V = inputs["u"].shape[1], inputs["u"].shape[3]
        h = torch.zeros((B, HV, _num_chunks(T, int(inputs["chunk_size"])), K, V), dtype=inputs["u"].dtype, device=inputs["u"].device)
        return h, torch.zeros_like(inputs["u"])
    B, _, T, K = inputs["q"].shape
    HV, V = inputs["dv"].shape[1], inputs["dv"].shape[3]
    dh = torch.zeros((B, HV, _num_chunks(T, int(inputs["chunk_size"])), K, V), dtype=inputs["dv"].dtype, device=inputs["dv"].device)
    return dh, torch.zeros_like(inputs["dv"])


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """运行 CPU 同精度或 fp64 高精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    return _zero_h_ref(inputs)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """运行 NPU DUT。"""
    inputs = build_inputs(spec, _marker_device(input_data), high_precision=False)
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_bwd_dhu(inputs["q"], inputs["k"], inputs["w"], inputs["do"], inputs["dv"], inputs["scale"], inputs["chunk_size"], g=inputs["g"], gK=None, h0=None, dht=None, cu_seqlens=None, chunk_indices=None, transpose_state_layout=False)


@register("executor_chunk_gated_delta_rule_bwd_dhu")
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
