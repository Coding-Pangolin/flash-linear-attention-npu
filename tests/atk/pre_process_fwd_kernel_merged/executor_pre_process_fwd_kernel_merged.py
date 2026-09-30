"""pre_process_fwd_kernel_merged of ATK executor.

本算子是 CP（context parallel）场景下 GDN / KDA / DPLR 前向的 pre-process 融合算子：
一次调用处理**一个打包窗口**（窗口内可含多段序列，段边界由 `cu_seqlens` 给出），
输出 `hm[Nseq, HV, K, V+K]`（左 `[0,V)` 为 `h`，右 `[V,V+K)` 为 `m`）。

输入布局：**BNSD `[B,H,T,D]`**（与仓内其它 AscendC 算子一致），`B ≡ 1`。
CPU 标杆：本目录 `scripts/pre_process_fwd_kernel_merged_cpu.py`
（纯 PyTorch，token-major `[T,H,D]`），本文件只做布局搬运与逐段调用。

精度口径（重要）：本算子的**验收基线是"契约版"标杆** —— `accum_dtype=fp32` +
三个舍入点开关全开。`reference.py` 的模块文档写明：kernel 的 h/m 累加器是 FP32，
用 FP64 基准会让任何忠实实现平白多出 ~9.4e-3 的绝对偏差（与 H20 `ieee` 对齐时实测）。
因此 **`high_precision=True` 只用于参考侧的灵敏度对照（ATK 的 benchmark 节点），
不作为本算子的验收真值**；验收真值走 `high_precision=False`。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
)


OP_NAME = "pre_process_fwd_kernel_merged"
ACLNN_NAME = "PreProcessFwdKernelMerged"

K_DIM = 128
V_DIM = 128
BT = 64

# CPU 标杆放在本算子 ATK 目录的 scripts/ 下（与仓内其它算子的约定一致）。
_REFERENCE_PY = (
    Path(__file__).resolve().parent / "scripts" / "pre_process_fwd_kernel_merged_cpu.py"
)


def _load_reference():
    """加载本目录 scripts/ 下的 CPU 标杆。"""
    if not _REFERENCE_PY.is_file():
        raise FileNotFoundError(f"找不到 CPU 标杆：{_REFERENCE_PY}")
    spec = importlib.util.spec_from_file_location("_ppfm_reference", _REFERENCE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ppfm_reference"] = module
    spec.loader.exec_module(module)
    return module.pre_process_fwd_kernel_merged


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """按 **token-major** 构造本算子的输入（与 `scripts/pre_process_fwd_kernel_merged_cpu.py` 同分布）。

    spec 字段：dtype / B(=1) / HK / HV / T / K(=128) / V(=128) / chunk_size(=64)
              / cu_seqlens(可选, list[int]) / gate(g|gk) / gate_dtype(fp32|bf16) / route / soc

    **数据分布必须是模型同构的**：
    `k` 归一化、`w = beta · k`（`beta ~ U(0, 0.02)`），使 `|Kw| << 1`、`m` 链良态。
    若改用满幅随机 `w`，`m = Π M_c` 会把 fp32 求和顺序的 1 ulp 差异放大到 O(1)
    （`|m| ~ 1e7`），那是**用例病态**、不是实现缺陷 —— 交付件里不允许出现这种用例。

    head 约定：`k` **恒为 `[T, HK, K]`**（即使 gk/GVA 路径，门控按 value head 给 `gk[T,HV,K]`）；
    `w/u/v` 在 `HV` 维。`HK` 与 `HV` 成倍数（`HV % HK == 0`）。
    """
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260818))
    elem = _orig_dtype(dtype_name)

    HK = int(spec.get("HK", 1))
    HV = int(spec.get("HV", 1))
    T = int(spec.get("T", 1024))
    K = int(spec.get("K", K_DIM))
    V = int(spec.get("V", V_DIM))
    chunk_size = int(spec.get("chunk_size", BT))
    gate_kind = str(spec.get("gate", "g")).lower()
    gate_dtype = str(spec.get("gate_dtype", "fp32")).lower()

    cu_raw = spec.get("cu_seqlens")
    cu = [int(x) for x in cu_raw] if cu_raw else [0, T]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def randn(shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def real_gate(shape_tail):
        """真实分布：每个 chunk 内做 cumsum 的负对数衰减（与标杆 self_test 同款）。"""
        nblk = -(-T // chunk_size)
        base = -0.013 / chunk_size * (1 + torch.rand(nblk, chunk_size, *shape_tail, generator=gen) * 0.5)
        return base.cumsum(1).reshape(nblk * chunk_size, *shape_tail)[:T].contiguous()

    # 模型同构数据：k 归一化、w = beta * k（beta ~ U(0, 0.02)），保证 |Kw| 远小于 1、m 链良态。
    k = torch.nn.functional.normalize(randn((T, HK, K)), dim=-1).to(elem)
    v = randn((T, HV, V)).to(elem)
    beta = torch.rand((T, HV, 1), generator=gen, dtype=torch.float32) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).to(elem)
    u = v.clone()

    inputs: dict[str, Any] = {
        "k": k.to(calc),
        "v": v.to(calc),
        "w": w.to(calc),
        "u": u.to(calc),
        "cu_seqlens": [int(x) for x in cu],
        "chunk_size": chunk_size,
        "gate_kind": gate_kind,
    }
    if gate_kind == "gk":
        # KDA：逐 K 门控，按 value head 给（HV 个），k 仍按 HK 头 ⇒ HK < HV（GVA）合法。
        inputs["gk"] = real_gate((HV, K)).to(elem if gate_dtype != "fp32" else torch.float32).to(calc)
        inputs["g"] = None
    else:
        inputs["g"] = real_gate((HV,)).to(elem if gate_dtype != "fp32" else torch.float32).to(calc)
        inputs["gk"] = None
    return inputs


def _to_bnsd(x: torch.Tensor) -> torch.Tensor:
    """token-major `[T,H,(D)]` -> DUT 要求的 BNSD `[1,H,T,(D)]`。"""
    return x.movedim(1, 0).unsqueeze(0).contiguous()


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """CPU 标杆：按 `cu_seqlens` **逐段**调用 reference，再 stack 成 `[Nseq, HV, K, V+K]`。

    与算子契约一一对应：算子的 `hm[i]` == 竞品/标杆对第 i 段单独调用一次的结果。
    """
    ref = _load_reference()
    inputs = build_inputs(spec, torch.device("cpu"), high_precision)
    cu = [int(x) for x in inputs["cu_seqlens"]]
    chunk_size = int(inputs["chunk_size"])

    # 契约版（验收基线）与高精度对照版：只有 accum_dtype 与三个舍入开关不同。
    if high_precision:
        accum_dtype = torch.float64
        round_h = round_vnew = round_affine = False
    else:
        accum_dtype = torch.float32
        round_h = round_vnew = round_affine = True

    outs = []
    for i in range(len(cu) - 1):
        outs.append(
            ref(
                inputs["k"], inputs["v"], inputs["w"],
                g=inputs["g"], gk=inputs["gk"], bg=None, u=inputs["u"],
                chunk_size=chunk_size,
                cu_seqlens=[cu[i], cu[i + 1]],
                accum_dtype=accum_dtype,
                round_h_to_input_dtype=round_h,
                round_v_new_to_input_dtype=round_vnew,
                round_affine_chain_to_float32=round_affine,
            )
        )
    return torch.stack(outs, dim=0)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """NPU DUT：BNSD 输入，调用仓内 `fla_npu.ops.ascendc.pre_process_fwd_kernel_merged`。"""
    dev = _marker_device(input_data)
    inputs = build_inputs(spec, dev, high_precision=False)
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    return npu_ppfm(
        _to_bnsd(inputs["k"]), _to_bnsd(inputs["w"]), _to_bnsd(inputs["u"]),
        v=None,
        g=(None if inputs["g"] is None else _to_bnsd(inputs["g"])),
        gk=(None if inputs["gk"] is None else _to_bnsd(inputs["gk"])),
        bg=None,
        cu_seqlens=list(inputs["cu_seqlens"]),
        chunk_size=int(inputs["chunk_size"]),
    )


@register("executor_pre_process_fwd_kernel_merged")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        elif self.device == "gpu":
            # 本算子没有 GPU 参考实现；GPU 节点只用于与 CPU 同一套标杆做加速。
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(
                f"{OP_NAME} needs an NPU DUT and a CPU reference, device={self.device!r}, "
                f"benchmark={self.is_benchmark_task}"
            )
        return _finite_tuple(outputs, golden=(self.device != "npu"))
