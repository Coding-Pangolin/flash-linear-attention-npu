"""bwd_dhu GVA 双标杆测试：随机输入直接测，无需 example dump。"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import ct
import fla_npu
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
CASES_JSON = os.path.join(_REPO_ROOT, "gpu", "cases.json")

_spec = importlib.util.spec_from_file_location("test_bwd_dhu_golden", os.path.join(_SCRIPT_DIR, "test_bwd_dhu.py"))
_golden_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden_mod)

chunk_gated_delta_rule_bwd_dhu_cpu = _golden_mod.chunk_gated_delta_rule_bwd_dhu_cpu
create_bwd_dhu_random_inputs = _golden_mod.create_bwd_dhu_random_inputs
effective_scale = _golden_mod.effective_scale
generate_cu_seqlens = _golden_mod.generate_cu_seqlens
prepare_chunk_indices = _golden_mod.prepare_chunk_indices
scale_for_compute_dtype = _golden_mod.scale_for_compute_dtype

DEFAULT_OUT_DIR = os.path.join(_SCRIPT_DIR, "bwd_dhu_out")

# gpu/cases.json 中 GPU dump 不支持、需 CPU dual 跑的 case
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


@dataclass
class BwdDhuCase:
    name: str
    batch: int
    k_h: int
    v_h: int
    tokens: int
    v_dim: int = 256
    k_dim: int = 128
    chunk_size: int = 64
    varlen: bool = True
    cu_seqlens_len: Optional[int] = None
    dtype: str = "bf16"
    gtype: str = "fp32"
    supported: bool = True
    skip_reason: str = ""

    def ktype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bf16" else torch.float16

    def gate_dtype(self) -> torch.dtype:
        if self.gtype == "bf16":
            return torch.bfloat16
        if self.gtype == "fp16":
            return torch.float16
        return torch.float32


# 与 fwd_h 12 项矩阵对齐，Vdim=256；按规模从小到大排列（去掉与 smoke_fixed 重复的 fixed_t4096）
CASES = [
    BwdDhuCase("smoke_varlen_t256_v256", 1, 16, 32, 256, chunk_size=64, cu_seqlens_len=5),
    BwdDhuCase("fixed_b176_t24_v256", 176, 2, 64, 24, chunk_size=64, varlen=False),
    BwdDhuCase("fixed_b711_t196_v256", 711, 4, 32, 196, chunk_size=128, varlen=False),
    BwdDhuCase("fixed_b16_t2048_v256", 16, 21, 63, 2048, chunk_size=64, varlen=False),
    BwdDhuCase("smoke_fixed_t4096_v256", 1, 16, 32, 4096, chunk_size=64, varlen=False),
    BwdDhuCase("varlen_t16384_v256_cu2", 1, 21, 63, 16384, chunk_size=64, cu_seqlens_len=2),
    BwdDhuCase("varlen_t16384_v256_cu128", 1, 16, 32, 16384, chunk_size=64, cu_seqlens_len=128),
    BwdDhuCase("varlen_t65536_v256_cu17", 1, 4, 32, 65536, chunk_size=128, cu_seqlens_len=17),
    BwdDhuCase("varlen_t65536_v256_cu172", 1, 8, 32, 65536, chunk_size=128, cu_seqlens_len=172),
    BwdDhuCase("varlen_t65536_v256_cu668", 1, 16, 32, 65536, chunk_size=64, cu_seqlens_len=668),
    BwdDhuCase(
        "varlen_t262144_v256_cu32", 1, 2, 64, 262144, chunk_size=64, cu_seqlens_len=32,
        supported=False,
        skip_reason="CPU fp64 golden T=262144 过慢，暂跳过",
    ),
]


def _normalize_dtype_str(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in ("bf16", "bfloat16"):
        return "bf16"
    if key in ("fp16", "float16"):
        return "fp16"
    return "fp32"


def _load_cases_json(path: str | None = None) -> dict[str, dict]:
    json_path = path or os.environ.get("BWD_HU_CASES_JSON", CASES_JSON)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("cases", data) if isinstance(data, dict) else data
    return {c["name"]: c for c in entries}


def case_from_json(entry: dict) -> BwdDhuCase:
    varlen = bool(entry.get("varlen", False))
    tokens = int(entry["T"])
    supported = True
    skip_reason = ""
    if tokens >= 262144:
        supported = False
        skip_reason = f"CPU fp64 golden T={tokens} 过慢，暂跳过"

    return BwdDhuCase(
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
        dtype=_normalize_dtype_str(entry.get("dtype", "bf16")),
        gtype=_normalize_dtype_str(entry.get("gtype", "fp32")),
        supported=supported,
        skip_reason=skip_reason,
    )


def load_cases_from_json(names: list[str], path: str | None = None) -> list[BwdDhuCase]:
    by_name = _load_cases_json(path)
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"unknown case(s) in cases.json: {', '.join(missing)}")
    return [case_from_json(by_name[n]) for n in names]


CASES_JSON_UNSUPPORTED = load_cases_from_json(CASES_JSON_UNSUPPORTED_NAMES)

# 内置矩阵 + cases.json 中 GPU 不支持的 case（按 name 去重，json 优先覆盖同名字段）
_BUILTIN_BY_NAME = {c.name: c for c in CASES}
for _jc in CASES_JSON_UNSUPPORTED:
    _BUILTIN_BY_NAME[_jc.name] = _jc
ALL_CASES = list(_BUILTIN_BY_NAME.values())


def _cu_seqlens_for_case(case: BwdDhuCase) -> list[int]:
    cu_len = case.cu_seqlens_len or 2
    if cu_len <= 2:
        return [0, case.tokens]
    batchsize = cu_len - 1
    seg_avg = (case.tokens + batchsize - 1) // batchsize
    seg_max = max(case.chunk_size, seg_avg, min(128, case.chunk_size * 2))
    if case.tokens > 8192:
        seg_max = max(seg_max, seg_avg)
    return generate_cu_seqlens(
        cu_len, case.tokens, seg_min=case.chunk_size, seg_max=seg_max,
    )


def _resolve_cases() -> list[BwdDhuCase]:
    suite = os.environ.get("BWD_HU_SUITE", "").strip().lower()
    only = os.environ.get("BWD_HU_CASE", "").strip()

    if suite in ("unsupported", "casesjson", "gpu_unsupported"):
        cases = list(CASES_JSON_UNSUPPORTED)
    elif suite in ("all", "full"):
        cases = list(ALL_CASES)
    else:
        cases = list(CASES)

    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        by_name = {c.name: c for c in ALL_CASES}
        selected: list[BwdDhuCase] = []
        missing: list[str] = []
        for name in names:
            if name in by_name:
                selected.append(by_name[name])
            else:
                missing.append(name)
        if missing:
            raise SystemExit(f"unknown BWD_HU_CASE: {', '.join(missing)}")
        cases = selected
    return cases


def _dual_check(name: str, npu_out: torch.Tensor, ref_fp64: torch.Tensor, ref_npu: torch.Tensor, level: str = "L1"):
    print(f"================== {name} (dual: fp64 gt / npu-aligned bench) ==================", flush=True)
    result = ct.dual(npu_out.cpu().float(), ref_fp64.cpu().float(), ref_npu.cpu().float(), level=level)
    ok = bool(result.get("success"))
    ratios = result.get("ratios", {})
    checks = result.get("checks", {})
    tag = "PASS" if ok else "FAIL"
    print(f"[{name}] dual {tag}: checks={checks} ratios={ratios}", flush=True)
    return ok, ratios, checks


def _build_inputs(case: BwdDhuCase, seed: int = 0):
    ktype = case.ktype()
    gtype = case.gate_dtype()
    torch.manual_seed(seed)
    q, k, w, do, dv, g = create_bwd_dhu_random_inputs(
        case.batch, case.k_h, case.v_h, case.tokens, case.k_dim, case.v_dim, ktype, gtype,
    )
    cu_seqlens = None
    chunk_indices = None
    if case.varlen:
        cu_seqlens = _cu_seqlens_for_case(case)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)
    scale = scale_for_compute_dtype(effective_scale(1.0 / math.sqrt(case.k_dim), case.k_dim), ktype)
    return q, k, w, do, dv, g, cu_seqlens, chunk_indices, scale


def run_case(case: BwdDhuCase, device: int, out_root: str, seed: int = 0) -> tuple[str, str]:
    if not case.supported:
        return "SKIP", case.skip_reason

    print(f"\n========== {case.name} ==========", flush=True)
    print(
        f"B={case.batch} Hk/Hv={case.k_h}/{case.v_h} T={case.tokens} "
        f"K={case.k_dim} V={case.v_dim} cs={case.chunk_size} varlen={case.varlen}",
        flush=True,
    )

    q, k, w, do, dv, g, cu_seqlens, chunk_indices, scale = _build_inputs(case, seed=seed)

    dh_fp64, _, dv2_fp64 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, do, dv, cu_seqlens, chunk_indices, g=g, scale=scale,
        chunk_size=case.chunk_size, golden_mode="fp64",
    )
    dh_npu_bench, _, dv2_npu_bench = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, do, dv, cu_seqlens, chunk_indices, g=g, scale=scale,
        chunk_size=case.chunk_size, golden_mode="npu",
    )

    dh_npu, _, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), w.npu(), do.npu(), dv.npu(),
        scale=scale,
        chunk_size=case.chunk_size,
        g=g.npu(),
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    dh_ok, _, _ = _dual_check("dh", dh_npu, dh_fp64, dh_npu_bench)
    dv2_ok, _, _ = _dual_check("dv2", dv2_npu, dv2_fp64, dv2_npu_bench)

    case_dir = os.path.join(out_root, case.name)
    if os.environ.get("BWD_HU_SAVE_OUT", "1") == "1":
        os.makedirs(case_dir, exist_ok=True)
        torch.save({
            "case": case.name,
            "dh_npu": dh_npu.cpu(),
            "dv2_npu": dv2_npu.cpu(),
            "dh_fp64": dh_fp64.cpu(),
            "dh_npu_bench": dh_npu_bench.cpu(),
            "dv2_fp64": dv2_fp64.cpu(),
            "dv2_npu_bench": dv2_npu_bench.cpu(),
        }, os.path.join(case_dir, "outputs.pt"))

    if dh_ok and dv2_ok:
        return "PASS", f"dh=PASS dv2=PASS | out={case_dir}"
    return "FAIL", f"dh={'PASS' if dh_ok else 'FAIL'} dv2={'PASS' if dv2_ok else 'FAIL'}"


def main():
    device = int(os.environ.get("TEST_DEVICE_ID", 5))
    torch.npu.set_device(device)
    out_root = os.environ.get("BWD_HU_OUT_DIR", DEFAULT_OUT_DIR)
    cases = _resolve_cases()

    print(f"[MODE] random input + dual(fp64 gt / npu-aligned bench), no example dump", flush=True)
    print(
        f"device={device} out_root={out_root} cases={len(cases)} "
        f"suite={os.environ.get('BWD_HU_SUITE', 'builtin')}",
        flush=True,
    )

    results = []
    for case in cases:
        try:
            status, detail = run_case(case, device, out_root)
        except Exception as exc:
            status, detail = "FAIL", str(exc)
            print(f"FAIL: {exc}", flush=True)
        results.append((case.name, status, detail))

    print("\n===== SUMMARY =====", flush=True)
    for name, status, detail in results:
        print(f"{name}: {status} ({detail})", flush=True)

    if any(s == "FAIL" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
