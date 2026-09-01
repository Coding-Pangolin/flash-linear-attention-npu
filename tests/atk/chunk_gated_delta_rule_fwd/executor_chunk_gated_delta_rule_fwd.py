#!/usr/bin/env python3
"""chunk_gated_delta_rule_fwd 的 ATK 原生双标杆执行器。"""

from __future__ import annotations

from typing import Any

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from atk_role_contract import role_for_atk_task
from gdn_reference import (
    GdnCase,
    canonical_chunk_indices,
    deterministic_initial_state,
    effective_inputs,
    mask_a_contract,
    output_names,
    run_golden_reference,
)
from six_aclnn_benchmark import (
    SIX_ACLNN_OPS,
    VARLEN_CUMSUM_TRANSPORT,
    run_six_aclnn_core,
)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _scalar(value: Any):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _bool(value: Any) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(value: Any) -> int:
    return int(_scalar(value))


def _float(value: Any) -> float:
    return float(_scalar(value))


def _text(value: Any) -> str:
    return str(_scalar(value)).strip()


def _cu_seqlens(specification: str, tokens: int, enabled: bool):
    if not enabled:
        return None
    values = tuple(int(value) for value in specification.split(",") if value.strip())
    if len(values) < 2 or values[0] != 0 or values[-1] != tokens:
        raise ValueError(
            f"cu_seqlens 必须从 0 开始并以 T={tokens} 结束：{values}"
        )
    return values


@register("executor_chunk_gated_delta_rule_fwd")
class ChunkGatedDeltaRuleFwdApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self._task_name = str(task_result.name or "")
        self._is_benchmark_task = bool(task_result.is_benchmark_task)
        self._case_id = int(task_result.case_config.id)
        self._case = None
        self._public_dtype = None
        self._inputs = None
        self._role = "uninitialized"
        self._forward = None
        self._output_names = ()
        self._execution_device_id = None

    def _npu_device(self, device_id: int):
        import torch_npu  # noqa: F401

        # ATK 的 NPU solo worker 已绑定 node.yaml 中的设备。执行器只构造
        # 显式 device 并让张量携带路由，不再修改进程级 current device。
        return torch.device(f"npu:{device_id}")

    def init_by_input_data(self, input_data: InputDataset):
        values = input_data.kwargs
        dtype_name = _text(values["qkv_dtype"]).lower()
        try:
            public_dtype = DTYPES[dtype_name]
        except KeyError as exc:
            raise ValueError(f"qkv_dtype 仅支持 {sorted(DTYPES)}，实际为 {dtype_name}") from exc

        q, k, v, g, beta = effective_inputs(
            values["q"],
            values["k"],
            values["v"],
            values["g"],
            values["beta"],
            public_dtype,
        )
        is_varlen = _bool(values["is_varlen"])
        scenario = _text(values["scenario"]).lower()
        cu_spec = _text(values["cu_seqlens_spec"])
        case = GdnCase(
            batch=q.shape[0],
            k_heads=q.shape[1],
            v_heads=v.shape[1],
            tokens=q.shape[2],
            key_dim=q.shape[3],
            value_dim=v.shape[3],
            chunk_size=_int(values["chunk_size"]),
            scale=_float(values["scale"]),
            scenario=scenario,
            cu_seqlens=_cu_seqlens(cu_spec, q.shape[2], is_varlen),
        )
        case.validate()
        if scenario == "dense" and is_varlen:
            raise ValueError("dense 场景不能设置 is_varlen=true")
        if scenario == "varlen" and not is_varlen:
            raise ValueError("varlen 场景必须设置 is_varlen=true")

        self._case = case
        self._public_dtype = public_dtype
        self._inputs = (q, k, v, g, beta)
        self._output_names = output_names(case)

        # 保存并复用三路完全一致的有效输入，而不是生成器的 raw g/beta。
        values["q"] = q
        values["k"] = k
        values["v"] = v
        values["g"] = g
        values["beta"] = beta

        self._role = role_for_atk_task(
            self.device,
            self._task_name,
            self._is_benchmark_task,
        )
        if self._role == "golden":
            print(
                f"[gdn-double-atk] case={self._case_id} "
                "role=golden target=cpu_fp64_recurrence",
                flush=True,
            )
            return

        device_id = int(self.device_id)
        self._execution_device_id = device_id
        device = self._npu_device(device_id)
        q_npu = q.to(device).contiguous()
        k_npu = k.to(device).contiguous()
        v_npu = v.to(device).contiguous()
        g_npu = g.to(device).contiguous()
        beta_npu = beta.to(device).contiguous()
        initial_state = deterministic_initial_state(case)
        if initial_state is not None:
            initial_state = initial_state.to(device).contiguous()

        if self._role == "benchmark":
            def benchmark_forward():
                from fla_npu.ops import ascendc

                missing = [name for name in SIX_ACLNN_OPS if not hasattr(ascendc, name)]
                if missing:
                    raise RuntimeError(f"当前 fla_npu 缺少六 ACLNN 接口：{missing}")
                outputs = run_six_aclnn_core(
                    ascendc,
                    q_npu,
                    k_npu,
                    v_npu,
                    g_npu,
                    beta_npu,
                    initial_state=initial_state,
                    output_final_state=case.output_final_state,
                    chunk_size=case.chunk_size,
                    cu_seqlens=case.cu_seqlens,
                    scale=case.scale,
                )
                o, final_state, g_cumsum, a = outputs
                if case.output_final_state:
                    if final_state is None:
                        raise RuntimeError("请求 final_state，但六 ACLNN benchmark 返回 None")
                    return o, final_state, g_cumsum, a
                return o, g_cumsum, a

            self._forward = benchmark_forward
            print(
                f"[gdn-double-atk] case={self._case_id} "
                "role=benchmark target=six_aclnn_npu "
                f"node={self._task_name} device={device_id} "
                f"ops={','.join(SIX_ACLNN_OPS)}",
                flush=True,
            )
            return

        cu_values = None if case.cu_seqlens is None else list(case.cu_seqlens)
        chunk_indices = canonical_chunk_indices(case.cu_seqlens, case.chunk_size)

        def dut_forward():
            from fla_npu.ops import ascendc

            if not hasattr(ascendc, "chunk_gated_delta_rule_fwd"):
                raise RuntimeError("当前 fla_npu 包未提供 chunk_gated_delta_rule_fwd")
            outputs = ascendc.chunk_gated_delta_rule_fwd(
                q_npu,
                k_npu,
                v_npu,
                g_npu,
                beta_npu,
                initial_state=initial_state,
                output_final_state=case.output_final_state,
                chunk_size=case.chunk_size,
                cu_seqlens=cu_values,
                chunk_indices=chunk_indices,
                scale=case.scale,
            )
            o, final_state, g_cumsum, a = outputs
            if case.output_final_state:
                if final_state is None:
                    raise RuntimeError("请求 final_state，但 NPU core 返回 None")
                return o, final_state, g_cumsum, a
            return o, g_cumsum, a

        self._forward = dut_forward
        print(
            f"[gdn-double-atk] case={self._case_id} "
            "role=dut target=chunk_gated_delta_rule_fwd "
            f"node={self._task_name} device={device_id}",
            flush=True,
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._case is None or self._inputs is None or self._public_dtype is None:
            raise RuntimeError("GDN ATK 执行器尚未初始化")
        with torch.no_grad():
            if self._role in {"dut", "benchmark"}:
                if self._forward is None:
                    raise RuntimeError(f"{self._role} forward 未初始化")
                outputs = self._forward()
            elif self._role == "golden":
                outputs = run_golden_reference(
                    *self._inputs,
                    self._case,
                    self._public_dtype,
                )
            else:
                raise RuntimeError(f"未知执行角色：{self._role}")

        if not with_output:
            return None
        normalized = []
        for index, output in enumerate(outputs):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"output[{index}] 不是 Tensor：{type(output)!r}")
            output = output.detach().cpu().contiguous()
            if self._output_names[index] == "A":
                # 在同步和回传后做 CPU 侧契约掩码，避免额外 NPU 算子污染 profile。
                output = mask_a_contract(output, self._case)
            if not bool(torch.isfinite(output.float()).all()):
                raise RuntimeError(f"output[{index}] 包含 NaN/Inf")
            normalized.append(output.contiguous())
        return normalized[0] if len(normalized) == 1 else tuple(normalized)

    def export_custom_data(self, *_args, **_kwargs):
        return {
            "output_names": list(self._output_names),
            "role": self._role,
            "target": {
                "dut": "chunk_gated_delta_rule_fwd",
                "benchmark": "six_aclnn_npu",
                "golden": "cpu_fp64_recurrence",
            }.get(self._role, "uninitialized"),
            "benchmark_ops": list(SIX_ACLNN_OPS) if self._role == "benchmark" else [],
            "varlen_cumsum_transport": (
                VARLEN_CUMSUM_TRANSPORT if self._role == "benchmark" else ""
            ),
            "benchmark_role_transport": "atk_named_npu_node",
            "execution_device_id": self._execution_device_id,
            "a_padding_policy": "zero_non_contract_tail",
        }
