"""fwd_h GVA 用例：先用 example dump 模型分布输入，再单算子精度比对。"""
import glob
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import ct
import fla_npu
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

_TEST_FWD_H = os.path.join(os.path.dirname(__file__), "test_fwd_h.py")
_spec = importlib.util.spec_from_file_location("test_fwd_h_golden", _TEST_FWD_H)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
forward_h_trans_cpu = _mod.forward_h_trans_cpu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
EXAMPLE = os.path.join(REPO_ROOT, "examples/flash_gated_delta_rule.py")
DEFAULT_DUMP_DIR = os.path.join(
    REPO_ROOT,
    "examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data",
)


def cdiv(a, b):
    return (a + b - 1) // b


@dataclass
class FwdHCase:
    name: str
    batch: int
    k_h: int
    v_h: int
    tokens: int
    v_dim: int
    k_dim: int = 128
    chunk_size: int = 64
    varlen: bool = True
    cu_seqlens_len: Optional[int] = None
    dtype: str = "bf16"
    supported: bool = True
    skip_reason: str = ""

    def mean_len(self) -> int:
        if not self.varlen:
            return self.tokens
        # cu_seqlens 长度 L => L-1 条序列
        num_seqs = max(1, int(self.cu_seqlens_len) - 1)
        return max(1, int(self.tokens // num_seqs))


CASES = [
  # 原 Vdim=256 用例统一改为 Vdim=128 跑
  FwdHCase("varlen_t16384_v128_cu128", 1, 16, 32, 16384, 128, chunk_size=64, cu_seqlens_len=128),
  FwdHCase("varlen_t16384_v128_cu2", 1, 21, 63, 16384, 128, chunk_size=64, cu_seqlens_len=2),
  FwdHCase("varlen_t65536_v128_cu172", 1, 8, 32, 65536, 128, chunk_size=128, cu_seqlens_len=172),
  FwdHCase("varlen_t65536_v128_cu668", 1, 16, 32, 65536, 128, chunk_size=64, cu_seqlens_len=668),
  FwdHCase("varlen_t65536_v128_cu17", 1, 4, 32, 65536, 128, chunk_size=128, cu_seqlens_len=17),
  FwdHCase("varlen_t262144_v128_cu32", 1, 2, 64, 262144, 128, chunk_size=64, cu_seqlens_len=32),
  FwdHCase("fixed_t4096_v128", 1, 16, 32, 4096, 128, chunk_size=64, varlen=False),
  FwdHCase("fixed_b16_t2048_v128", 16, 21, 63, 2048, 128, chunk_size=64, varlen=False),
  FwdHCase("fixed_b711_t196_v128", 711, 4, 32, 196, 128, chunk_size=128, varlen=False,
           supported=False, skip_reason="example 前置 Triton chunk_scaled_dot_kkt B=711 UB overflow，无法 dump"),
  FwdHCase("fixed_b176_t24_v128", 176, 2, 64, 24, 128, chunk_size=64, varlen=False),
  # 冒烟用小 shape（已有 dump 时可 FWD_H_TEST_ONLY=1 快速验证）
  FwdHCase("smoke_fixed_t4096_v128", 1, 16, 32, 4096, 128, chunk_size=64, varlen=False),
  FwdHCase("smoke_varlen_t256_v128", 1, 16, 32, 256, 128, chunk_size=64, cu_seqlens_len=5),
]


def dump_case(case: FwdHCase, device: int, dump_dir: str) -> str:
    pattern = os.path.join(dump_dir, f"{case.name}_*_cs{case.chunk_size}.pt")
    if os.environ.get("FWD_H_FORCE_DUMP", "0") != "1":
        matches = sorted(glob.glob(pattern))
        if matches:
            print(f"[DUMP] reuse existing {matches[-1]}", flush=True)
            return matches[-1]

    env = os.environ.copy()
    env["GDN_FWD_H_DUMP_DIR"] = dump_dir
    env["GDN_FWD_H_DUMP_NAME"] = case.name
    env["GDN_FWD_H_DUMP_EXIT"] = "1"
    cmd = [
        sys.executable,
        EXAMPLE,
        "--device", str(device),
        "--batch", str(case.batch),
        "--tokens", str(case.tokens),
        "--query-heads", str(case.k_h),
        "--value-heads", str(case.v_h),
        "--key-dim", str(case.k_dim),
        "--value-dim", str(case.v_dim),
        "--chunk-size", str(case.chunk_size),
        "--dtype", case.dtype,
    ]
    if case.varlen:
        cmd.extend(["--mean-len", str(case.mean_len())])
    else:
        cmd.append("--no-varlen")
    print(f"[DUMP] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"dump not found: {pattern}")
    return matches[-1]


def _assert_close(name: str, real: torch.Tensor, expect: torch.Tensor, diff_thd: float):
    print(f"================== {name} ==================", flush=True)
    out = ct.isclose(real.cpu(), expect, diff_thd=diff_thd, pct_thd=0.999)
    if out["result"] != "success":
        raise AssertionError(
            f"{name} compare failed: fulfill={out['fulfill_percent']:.4f}% "
            f"max_error={out['max_error']}"
        )


def test_dump(dump_path: str, device: int, diff_thd: float = 0.0001):
    torch.npu.set_device(device)
    data = torch.load(dump_path, map_location="cpu")
    chunk_size = int(data["chunk_size"])
    dtype = torch.bfloat16 if data.get("dtype", "bfloat16") == "bfloat16" else torch.float16

    k = data["k"].to(dtype)
    w = data["w"].to(dtype)
    u = data["u"].to(dtype)
    g = data["g"].float()
    cu_list = data.get("cu_seqlens")
    chunk_indices = data.get("chunk_indices")
    initial_state = data.get("initial_state")

    print(
        f"[INPUT SHAPE] dump={os.path.basename(dump_path)} "
        f"k={tuple(k.shape)} w={tuple(w.shape)} u={tuple(u.shape)} g={tuple(g.shape)} "
        f"chunk={chunk_size} varlen={cu_list is not None}",
        flush=True,
    )
    if cu_list is not None:
        print(f"[VARLEN] cu_seqlens_len={len(cu_list)} chunk_indices_len={len(chunk_indices)}", flush=True)

    cu_tensor = None if cu_list is None else torch.tensor(cu_list, dtype=torch.int64)
    ref_h, ref_v, _ = forward_h_trans_cpu(
        k, w, u, g,
        initial_state=initial_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_tensor,
    )

    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g=g.npu(),
        gk=None,
        initial_state=initial_state.npu() if initial_state is not None else None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_list,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()

    _assert_close("h", h, ref_h, diff_thd)
    _assert_close("v_new", v_new, ref_v, diff_thd)


def main():
    device = int(os.environ.get("TEST_DEVICE_ID", "5"))
    dump_dir = os.environ.get("GDN_FWD_H_DUMP_DIR", DEFAULT_DUMP_DIR)
    only = os.environ.get("FWD_H_CASE", "")
    dump_only = os.environ.get("FWD_H_DUMP_ONLY", "0") == "1"
    test_only = os.environ.get("FWD_H_TEST_ONLY", "0") == "1"

    torch.npu.set_device(device)
    os.makedirs(dump_dir, exist_ok=True)
    results = []
    for case in CASES:
        if only and case.name != only:
            continue
        print(f"\n========== {case.name} ==========", flush=True)
        if not case.supported:
            print(f"SKIP: {case.skip_reason}", flush=True)
            results.append((case.name, "SKIP", case.skip_reason))
            continue
        try:
            dump_path = os.path.join(dump_dir, f"{case.name}_0_cs{case.chunk_size}.pt")
            if not test_only:
                dump_path = dump_case(case, device, dump_dir)
            elif not os.path.isfile(dump_path):
                alt = sorted(glob.glob(os.path.join(dump_dir, f"{case.name}_*_cs{case.chunk_size}.pt")))
                alias = {
                    "varlen_t16384_v128_cu128": "fix_k16v32_t16384_*_cs64.pt",
                    "smoke_fixed_t4096_v128": "smoke_k16v32_t4096_*_cs64.pt",
                    "smoke_varlen_t256_v128": "smoke_varlen_t256_*_cs64.pt",
                    "fixed_t4096_v128": "smoke_k16v32_t4096_*_cs64.pt",
                }.get(case.name)
                if not alt and alias:
                    alt = sorted(glob.glob(os.path.join(dump_dir, alias)))
                if not alt:
                    raise FileNotFoundError(dump_path)
                dump_path = alt[-1]
            if dump_only:
                results.append((case.name, "DUMP_OK", dump_path))
                continue
            test_dump(dump_path, device)
            results.append((case.name, "PASS", dump_path))
        except Exception as exc:
            print(f"FAIL: {exc}", flush=True)
            results.append((case.name, "FAIL", str(exc)))

    print("\n===== SUMMARY =====", flush=True)
    for name, status, detail in results:
        print(f"{name}: {status} ({detail})", flush=True)

    if any(s == "FAIL" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
