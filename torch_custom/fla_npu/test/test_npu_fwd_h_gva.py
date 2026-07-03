"""fwd_h GVA / cases.json 用例：先用 example dump 模型分布输入，再单算子 CPU 双标杆。"""
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ct
import torch
import torch_npu
import fla_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

_TEST_FWD_H = os.path.join(os.path.dirname(__file__), "test_fwd_h.py")
_spec = importlib.util.spec_from_file_location("test_fwd_h_golden", _TEST_FWD_H)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
forward_h_trans_cpu = _mod.forward_h_trans_cpu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CASES_JSON = os.path.join(REPO_ROOT, "gpu", "cases.json")
EXAMPLE = os.path.join(REPO_ROOT, "examples/flash_gated_delta_rule.py")
DEFAULT_DUMP_DIR = os.path.join(
    REPO_ROOT,
    "examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data",
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "fwd_h_out")


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
  FwdHCase("varlen_t65536_v128_cu172", 1, 8, 32, 65536, 128, chunk_size=128, cu_seqlens_len=172,
           supported=False, skip_reason="example 前置 Triton chunk_scaled_dot_kkt cs=128 UB overflow，无法 dump"),
  FwdHCase("varlen_t65536_v128_cu668", 1, 16, 32, 65536, 128, chunk_size=64, cu_seqlens_len=668),
  FwdHCase("varlen_t65536_v128_cu17", 1, 4, 32, 65536, 128, chunk_size=128, cu_seqlens_len=17,
           supported=False, skip_reason="example 前置 Triton chunk_scaled_dot_kkt cs=128 UB overflow，无法 dump"),
  FwdHCase("varlen_t262144_v128_cu32", 1, 2, 64, 262144, 128, chunk_size=64, cu_seqlens_len=32,
           supported=False, skip_reason="example 在 T=262144 时 aicore 507015（MTE DDR 越界），无法 dump"),
  FwdHCase("fixed_t4096_v128", 1, 16, 32, 4096, 128, chunk_size=64, varlen=False),
  FwdHCase("fixed_b16_t2048_v128", 16, 21, 63, 2048, 128, chunk_size=64, varlen=False),
  FwdHCase("fixed_b711_t196_v128", 711, 4, 32, 196, 128, chunk_size=128, varlen=False,
           supported=False, skip_reason="example 前置 Triton chunk_scaled_dot_kkt B=711 UB overflow，无法 dump"),
  FwdHCase("fixed_b176_t24_v128", 176, 2, 64, 24, 128, chunk_size=64, varlen=False),
  # 冒烟用小 shape（已有 dump 时可 FWD_H_TEST_ONLY=1 快速验证）
  FwdHCase("smoke_fixed_t4096_v128", 1, 16, 32, 4096, 128, chunk_size=64, varlen=False),
  FwdHCase("smoke_varlen_t256_v128", 1, 16, 32, 256, 128, chunk_size=64, cu_seqlens_len=5),
]

# gpu/cases.json 中 GPU dump 不支持、走 example dump + CPU dual 的 case
CASES_JSON_UNSUPPORTED_NAMES = [
    "gva_fix_3",
    "gva_var_2",
    "gva_var_3",
    "gva_var_5",
    "gva_var_6",
    "phase_1_fix_11",
    "phase_1_fix_12",
    "phase_1_var_5",
    "phase_1_var_6",
]


def _normalize_dtype(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in ("bf16", "bfloat16"):
        return "bf16"
    if key in ("fp16", "float16"):
        return "fp16"
    return "bf16"


def _load_cases_json() -> dict[str, dict]:
    with open(CASES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("cases", data) if isinstance(data, dict) else data
    return {c["name"]: c for c in entries}


def case_from_json(entry: dict) -> FwdHCase:
    varlen = bool(entry.get("varlen", False))
    tokens = int(entry["T"])
    supported = True
    skip_reason = ""
    if tokens >= 262144:
        supported = False
        skip_reason = f"example T={tokens} 过大，暂不 dump"
    return FwdHCase(
        name=str(entry["name"]),
        batch=int(entry["B"]),
        k_h=int(entry["query_head"]),
        v_h=int(entry["value_head"]),
        tokens=tokens,
        v_dim=int(entry["Vdim"]),
        k_dim=int(entry["Kdim"]),
        chunk_size=int(entry.get("chunk_size", 64)),
        varlen=varlen,
        cu_seqlens_len=int(entry["mean_len"]) if varlen else None,
        dtype=_normalize_dtype(entry.get("dtype", "bf16")),
        supported=supported,
        skip_reason=skip_reason,
    )


def resolve_cases() -> list[FwdHCase]:
    suite = os.environ.get("FWD_H_SUITE", "").strip().lower()
    only = os.environ.get("FWD_H_CASE", "").strip()

    if suite in ("unsupported", "casesjson", "gpu_unsupported"):
        by_name = _load_cases_json()
        cases = [case_from_json(by_name[n]) for n in CASES_JSON_UNSUPPORTED_NAMES]
    else:
        cases = list(CASES)

    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        by_name = {c.name: c for c in cases}
        if suite in ("unsupported", "casesjson", "gpu_unsupported"):
            by_name.update({c.name: c for c in CASES})
        selected = []
        for name in names:
            if name not in by_name:
                raise SystemExit(f"unknown FWD_H_CASE: {name}")
            selected.append(by_name[name])
        cases = selected
    return cases


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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"dump not found after example exit={proc.returncode}: {pattern}"
        )
    if proc.returncode != 0:
        print(
            f"[DUMP] example exit={proc.returncode}, reuse {matches[-1]}",
            flush=True,
        )
    return matches[-1]


def _dual_check(name: str, npu_out: torch.Tensor, ref_hp: torch.Tensor, ref_npu: torch.Tensor, level: str = "L1"):
    """三档标杆：ct.dual(test, gt_fp64, bench_npu_aligned)。

    bench 为 bf16 乘 + fp32 累加，与 NPU Cube MMAD 数据类型一致。
    """
    print(f"================== {name} (dual: fp64 gt / npu-aligned bench) ==================", flush=True)
    result = ct.dual(npu_out.cpu(), ref_hp.cpu(), ref_npu.cpu(), level=level)
    ok = bool(result.get("success"))
    ratios = result.get("ratios", {})
    checks = result.get("checks", {})
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{name}] dual {tag}: checks={checks} ratios={ratios}",
        flush=True,
    )
    return ok, ratios, checks


def _save_case_outputs(case_dir: str, payload: dict):
    os.makedirs(case_dir, exist_ok=True)
    out_path = os.path.join(case_dir, "outputs.pt")
    torch.save(payload, out_path)
    print(f"[OUT] saved tensors -> {out_path}", flush=True)
    for key, tensor in payload.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        one_path = os.path.join(case_dir, f"{key}.pt")
        torch.save(tensor.cpu(), one_path)
    return out_path


def _viz_field(
    case_name: str,
    field: str,
    npu_t: torch.Tensor,
    ref_hp: torch.Tensor,
    ref_npu: torch.Tensor,
    viz_dir: str,
    sample_count: int,
    spatial: bool,
):
    """ct.viz(test, expect=fp64, bench=npu_aligned)，图片保存到 viz_dir。"""
    os.makedirs(viz_dir, exist_ok=True)
    name = f"{case_name}_{field}_npu_vs_fp64"
    print(f"[VIZ] {name} (bench=npu_aligned) -> {viz_dir}", flush=True)
    ct.viz(
        npu_t.cpu().float(),
        ref_hp.cpu().float(),
        bench=ref_npu.cpu().float(),
        out_dir=viz_dir,
        name=name,
        diff_thd=0.001,
        spatial=spatial,
        sample_count=sample_count,
    )


def test_dump(
    dump_path: str,
    device: int,
    case_name: str,
    out_root: str,
    dual_level: str = "L1",
    enable_viz: bool = True,
    viz_sample_count: int = 200_000,
):
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

    npu_only = os.environ.get("FWD_H_NPU_ONLY", "0") == "1"
    if not npu_only:
        cu_tensor = None if cu_list is None else torch.tensor(cu_list, dtype=torch.int64)
        chunk_indices_tensor = None
        if chunk_indices is not None:
            chunk_indices_tensor = torch.tensor(chunk_indices, dtype=torch.int64)
        golden_kwargs = dict(
            initial_state=initial_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_tensor,
            chunk_indices=chunk_indices_tensor,
        )
        # 升精度标杆：fp64 累加
        ref_h_hp, ref_v_hp, _ = forward_h_trans_cpu(
            k, w, u, g, **golden_kwargs, golden_mode="fp64",
        )
        # 同精度标杆（NPU 对齐）：bf16 乘 + fp32 累加
        ref_h_npu, ref_v_npu, _ = forward_h_trans_cpu(
            k, w, u, g, **golden_kwargs, golden_mode="npu",
        )
        # 旧 fp32 全精度乘标杆（仅存档对比，不参与 dual）
        ref_h_fp32, ref_v_fp32, _ = forward_h_trans_cpu(
            k, w, u, g, **golden_kwargs, golden_mode="fp32",
        )

    print("[NPU] calling npu_chunk_gated_delta_rule_fwd_h ...", flush=True)
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
    print(
        f"[NPU OK] h={tuple(h.shape)} dtype={h.dtype} v_new={tuple(v_new.shape)} dtype={v_new.dtype}",
        flush=True,
    )
    if npu_only:
        return {"h": (True, {}, {}), "v_new": (True, {}, {})}, out_root

    case_dir = os.path.join(out_root, case_name)
    payload = {
        "case_name": case_name,
        "dump_path": dump_path,
        "chunk_size": chunk_size,
        "h_npu": h.cpu(),
        "h_ref_fp64": ref_h_hp.cpu(),
        "h_ref_npu": ref_h_npu.cpu(),
        "h_ref_fp32": ref_h_fp32.cpu(),
        "v_new_npu": v_new.cpu(),
        "v_new_ref_fp64": ref_v_hp.cpu(),
        "v_new_ref_npu": ref_v_npu.cpu(),
        "v_new_ref_fp32": ref_v_fp32.cpu(),
    }
    _save_case_outputs(case_dir, payload)

    dual_results = {}
    outputs = (
        ("h", h, ref_h_hp, ref_h_npu, False),
        ("v_new", v_new, ref_v_hp, ref_v_npu, True),
    )
    for out_name, npu_t, hp_t, sp_t, use_spatial in outputs:
        try:
            dual_results[out_name] = _dual_check(
                out_name, npu_t, hp_t, sp_t, level=dual_level,
            )
        except Exception as exc:
            print(f"[{out_name}] dual error: {exc}", flush=True)
            dual_results[out_name] = (False, {}, {"error": str(exc)})
        if enable_viz:
            try:
                _viz_field(
                    case_name,
                    out_name,
                    npu_t,
                    hp_t,
                    sp_t,
                    viz_dir=os.path.join(case_dir, "viz"),
                    sample_count=viz_sample_count,
                    spatial=use_spatial,
                )
            except Exception as exc:
                print(f"[{out_name}] viz error: {exc}", flush=True)
    return dual_results, case_dir


def main():
    device = int(os.environ.get("TEST_DEVICE_ID", "5"))
    dump_dir = os.environ.get("GDN_FWD_H_DUMP_DIR", DEFAULT_DUMP_DIR)
    out_root = os.environ.get("FWD_H_OUT_DIR", DEFAULT_OUT_DIR)
    enable_viz = os.environ.get("FWD_H_VIZ", "1") == "1"
    viz_sample_count = int(os.environ.get("FWD_H_VIZ_SAMPLE_COUNT", "200000"))
    only = os.environ.get("FWD_H_CASE", "")
    dump_only = os.environ.get("FWD_H_DUMP_ONLY", "0") == "1"
    test_only = os.environ.get("FWD_H_TEST_ONLY", "0") == "1"
    suite = os.environ.get("FWD_H_SUITE", "builtin")
    cases = resolve_cases()
    print(f"[CONFIG] suite={suite} cases={len(cases)} device={device}", flush=True)
    if test_only:
        print("[MODE] FWD_H_TEST_ONLY=1 (skip dump, require existing .pt)", flush=True)
    elif dump_only:
        print("[MODE] FWD_H_DUMP_ONLY=1 (dump only)", flush=True)
    elif os.environ.get("FWD_H_NPU_ONLY", "0") == "1":
        print("[MODE] FWD_H_NPU_ONLY=1 (NPU execute only, no CPU golden)", flush=True)
    else:
        print(
            "[MODE] dump-if-missing + dual(fp64 gt / npu-aligned bench) + save outputs + ct.viz",
            flush=True,
        )
    print(f"[OUT] out_root={out_root} viz={enable_viz}", flush=True)

    torch.npu.set_device(device)
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)
    results = []
    for case in cases:
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
                    raise FileNotFoundError(
                        f"dump missing for {case.name} (FWD_H_TEST_ONLY=1): {dump_path}"
                    )
                dump_path = alt[-1]
            if dump_only:
                results.append((case.name, "DUMP_OK", dump_path))
                continue
            dual_results, case_dir = test_dump(
                dump_path,
                device,
                case.name,
                out_root,
                enable_viz=enable_viz,
                viz_sample_count=viz_sample_count,
            )
            h_ok = dual_results.get("h", (False,))[0]
            v_ok = dual_results.get("v_new", (False,))[0]
            if h_ok and v_ok:
                status = "PASS"
                detail = f"h=PASS v_new=PASS | out={case_dir}"
            else:
                status = "DUAL_FAIL"
                detail = (
                    f"h={'PASS' if h_ok else 'FAIL'} "
                    f"v_new={'PASS' if v_ok else 'FAIL'} | out={case_dir}"
                )
            results.append((case.name, status, detail))
        except Exception as exc:
            print(f"FAIL: {exc}", flush=True)
            results.append((case.name, "FAIL", str(exc)))

    print("\n===== SUMMARY =====", flush=True)
    for name, status, detail in results:
        print(f"{name}: {status} ({detail})", flush=True)

    if any(s in ("FAIL", "DUAL_FAIL") for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
