#!/usr/bin/env python3
"""GPU (Triton) vs CPU (fp64 golden) dual benchmark for chunk_kda_fwd_intra_sub_chunk.

Usage:
    # On GPU machine with Triton:
    python tests/kda_intra_sub_chunk_dual/test_gpu_vs_cpu.py

    # Specify output directory:
    KDA_ISC_OUT_DIR=./kda_isc_out python tests/kda_intra_sub_chunk_dual/test_gpu_vs_cpu.py

    # Run a single case:
    KDA_ISC_CASE=dense_B1_H2_T64_K32 python tests/kda_intra_sub_chunk_dual/test_gpu_vs_cpu.py

Outputs:
    - Per-case .pt files with all inputs, GPU outputs, CPU golden outputs
    - ct.viz charts (if ct is available) comparing GPU Aqk/Akkd vs CPU golden
    - Summary table
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_OPS_DIR = os.path.join(REPO_ROOT, "gpu")
sys.path.insert(0, GPU_OPS_DIR)
sys.path.insert(0, REPO_ROOT)

# Import CPU golden
from tests.kda_intra_sub_chunk_dual.cpu_golden import chunk_kda_fwd_intra_sub_chunk_ref, BC

# ---------------------------------------------------------------------------
# Try import ct for viz/dual
# ---------------------------------------------------------------------------
try:
    import ct
    HAS_CT = True
except ImportError:
    HAS_CT = False
    print("[WARN] ct not installed, skipping ct.viz / ct.dual charts")

# ---------------------------------------------------------------------------
# Try import GPU triton kernel
# ---------------------------------------------------------------------------
try:
    import triton
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_intra_sub_chunk
    from fla.ops.utils.op import exp2
    from fla.utils import IS_GATHER_SUPPORTED
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[WARN] Cannot import GPU Triton kernel (no triton / no CUDA). Will only run CPU golden.")


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
@dataclass
class IntraSubChunkCase:
    name: str
    B: int
    H: int
    T: int
    K: int
    chunk_size: int = 64
    dtype: torch.dtype = torch.bfloat16
    varlen: bool = False
    cu_seqlens_len: Optional[int] = None


CASES = [
    # Small smoke tests
    IntraSubChunkCase("dense_B1_H2_T64_K32", B=1, H=2, T=64, K=32, chunk_size=64),
    IntraSubChunkCase("dense_B1_H2_T128_K128", B=1, H=2, T=128, K=128, chunk_size=64),
    IntraSubChunkCase("dense_B2_H4_T64_K64", B=2, H=4, T=64, K=64, chunk_size=64),
    # Chunk size 32
    IntraSubChunkCase("dense_B1_H2_T64_K64_cs32", B=1, H=2, T=64, K=64, chunk_size=32),
    # Model-like shape
    IntraSubChunkCase("model_B1_H32_T512_K128", B=1, H=32, T=512, K=128, chunk_size=64),
    # Tail block (T not divisible by chunk_size)
    IntraSubChunkCase("tail_B1_H2_T100_K64", B=1, H=2, T=100, K=64, chunk_size=64),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices: list[int] = []
    for seq in range(int(cu_seqlens.numel()) - 1):
        length = int(cu_seqlens[seq + 1] - cu_seqlens[seq])
        n_chunks = (length + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            indices.extend([seq, local])
    return torch.tensor(indices, dtype=torch.long, device=cu_seqlens.device)


def make_inputs(case: IntraSubChunkCase, device: torch.device, seed: int = 42):
    torch.manual_seed(seed)
    B, H, T, K = case.B, case.H, case.T, case.K

    # BNSD layout for CPU golden: [B, H, T, K]
    q_bnsd = torch.randn(B, H, T, K, dtype=torch.float64, device="cpu")
    k_bnsd = torch.randn(B, H, T, K, dtype=torch.float64, device="cpu")
    # g: decreasing cumsum-like gates in log2 space
    g_bnsd = -torch.linspace(0, 40, T).view(1, 1, T, 1).expand(B, H, T, K).contiguous().double()
    beta_bnsd = torch.rand(B, H, T, dtype=torch.float64, device="cpu")
    scale = 1.0 / math.sqrt(K)

    return q_bnsd, k_bnsd, g_bnsd, beta_bnsd, scale


def bnsd_to_bsnd(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, K] -> [B, T, H, K]"""
    return x.permute(0, 2, 1, 3).contiguous()


def bsnd_to_bnsd(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H, K] -> [B, H, T, K]"""
    return x.permute(0, 2, 1, 3).contiguous()


def run_gpu_intra_sub_chunk(
    q_bsnd: torch.Tensor,
    k_bsnd: torch.Tensor,
    g_bsnd: torch.Tensor,
    beta_bsnd: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GPU Triton kernel. Inputs/outputs in BSND [B, T, H, K]."""
    B, T, H, K = q_bsnd.shape
    HV = H  # MHA, no GVA
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)

    Aqk = torch.empty(B, T, HV, BT, device=q_bsnd.device, dtype=q_bsnd.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=q_bsnd.device, dtype=torch.float32)

    grid = (NT, NC, B * HV)
    chunk_kda_fwd_kernel_intra_sub_chunk[grid](
        q=q_bsnd,
        k=k_bsnd,
        g=g_bsnd,
        beta=beta_bsnd,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        IS_VARLEN=False,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    return Aqk, Akkd


def run_case(case: IntraSubChunkCase, device: torch.device, out_dir: str) -> tuple[str, str]:
    print(f"\n{'='*60}")
    print(f"  {case.name}")
    print(f"  B={case.B} H={case.H} T={case.T} K={case.K} cs={case.chunk_size} dtype={case.dtype}")
    print(f"{'='*60}")

    q_bnsd, k_bnsd, g_bnsd, beta_bnsd, scale = make_inputs(case, device)

    # -----------------------------------------------------------------------
    # CPU golden (fp64, BNSD)
    # -----------------------------------------------------------------------
    cpu_aqk, cpu_akkd = chunk_kda_fwd_intra_sub_chunk_ref(
        q_bnsd, k_bnsd, g_bnsd, beta_bnsd, scale, case.chunk_size,
    )
    print(f"  CPU golden: aqk shape={cpu_aqk.shape} akkd shape={cpu_akkd.shape}")
    print(f"  CPU aqk abs mean={cpu_aqk.abs().mean().item():.6e}, akkd abs mean={cpu_akkd.abs().mean().item():.6e}")

    if not HAS_GPU:
        print("  [SKIP GPU] no Triton available")
        return "CPU-ONLY", "GPU not available"

    # -----------------------------------------------------------------------
    # GPU Triton (BSND)
    # -----------------------------------------------------------------------
    q_bsnd = bnsd_to_bsnd(q_bnsd.to(case.dtype).to(device))
    k_bsnd = bnsd_to_bsnd(k_bnsd.to(case.dtype).to(device))
    g_bsnd = bnsd_to_bsnd(g_bnsd.to(case.dtype).to(device))
    # beta: [B, H, T] -> [B, T, H]
    beta_bsnd = beta_bnsd.to(case.dtype).to(device).permute(0, 2, 1).contiguous()

    gpu_aqk_bsnd, gpu_akkd_bsnd = run_gpu_intra_sub_chunk(
        q_bsnd, k_bsnd, g_bsnd, beta_bsnd, scale, case.chunk_size,
    )
    # Convert GPU outputs to BNSD for comparison
    # GPU Aqk: [B, T, HV, BT] -> [B, HV, T, BT]
    gpu_aqk = gpu_aqk_bsnd.permute(0, 2, 1, 3).contiguous().float().cpu()
    # GPU Akkd: [B, T, HV, BC] -> [B, HV, T, BC]
    gpu_akkd = gpu_akkd_bsnd.permute(0, 2, 1, 3).contiguous().float().cpu()

    cpu_aqk_f32 = cpu_aqk.float()
    cpu_akkd_f32 = cpu_akkd.float()

    print(f"  GPU aqk abs mean={gpu_aqk.abs().mean().item():.6e}, akkd abs mean={gpu_akkd.abs().mean().item():.6e}")

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    aqk_diff = (gpu_aqk - cpu_aqk_f32).abs()
    akkd_diff = (gpu_akkd - cpu_akkd_f32).abs()
    aqk_max_diff = aqk_diff.max().item()
    akkd_max_diff = akkd_diff.max().item()
    aqk_mean_diff = aqk_diff.mean().item()
    akkd_mean_diff = akkd_diff.mean().item()

    print(f"  Aqk  max_diff={aqk_max_diff:.6e} mean_diff={aqk_mean_diff:.6e}")
    print(f"  Akkd max_diff={akkd_max_diff:.6e} mean_diff={akkd_mean_diff:.6e}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    case_dir = os.path.join(out_dir, case.name)
    os.makedirs(case_dir, exist_ok=True)
    torch.save({
        "case": case.name,
        "q_bnsd": q_bnsd.cpu(),
        "k_bnsd": k_bnsd.cpu(),
        "g_bnsd": g_bnsd.cpu(),
        "beta_bnsd": beta_bnsd.cpu(),
        "scale": scale,
        "chunk_size": case.chunk_size,
        "gpu_aqk": gpu_aqk.cpu(),
        "gpu_akkd": gpu_akkd.cpu(),
        "cpu_aqk": cpu_aqk_f32.cpu(),
        "cpu_akkd": cpu_akkd_f32.cpu(),
    }, os.path.join(case_dir, "outputs.pt"))

    # -----------------------------------------------------------------------
    # ct.viz (if available)
    # -----------------------------------------------------------------------
    if HAS_CT:
        try:
            ct.viz(gpu_aqk.numpy(), cpu_aqk_f32.numpy(),
                   out_dir=case_dir, name=f"aqk_{case.name}")
            ct.viz(gpu_akkd.numpy(), cpu_akkd_f32.numpy(),
                   out_dir=case_dir, name=f"akkd_{case.name}")
            print(f"  ct.viz charts saved to {case_dir}")
        except Exception as e:
            print(f"  ct.viz failed: {e}")

    # -----------------------------------------------------------------------
    # ct.dual (if available) — GPU as test, CPU fp64 as gt, CPU fp32 as bench
    # -----------------------------------------------------------------------
    dual_results = {}
    if HAS_CT:
        # For dual we need 3 things: test, gt(fp64), bench(same-precision)
        # Here GPU is 'test', CPU fp64 is 'gt', and we use CPU in input dtype as 'bench'
        cpu_aqk_bench = chunk_kda_fwd_intra_sub_chunk_ref(
            q_bnsd.to(case.dtype).double(), k_bnsd.to(case.dtype).double(),
            g_bnsd.to(case.dtype).double(), beta_bnsd.to(case.dtype).double(),
            scale, case.chunk_size,
        )[0].float()
        cpu_akkd_bench = chunk_kda_fwd_intra_sub_chunk_ref(
            q_bnsd.to(case.dtype).double(), k_bnsd.to(case.dtype).double(),
            g_bnsd.to(case.dtype).double(), beta_bnsd.to(case.dtype).double(),
            scale, case.chunk_size,
        )[1].float()
        try:
            r_aqk = ct.dual(gpu_aqk.numpy(), cpu_aqk_f32.numpy(), cpu_aqk_bench.numpy(), level="L1")
            r_akkd = ct.dual(gpu_akkd.numpy(), cpu_akkd_f32.numpy(), cpu_akkd_bench.numpy(), level="L1")
            dual_results["aqk"] = r_aqk
            dual_results["akkd"] = r_akkd
            aqk_ok = bool(r_aqk.get("success"))
            akkd_ok = bool(r_akkd.get("success"))
            print(f"  ct.dual Aqk: {'PASS' if aqk_ok else 'FAIL'} {r_aqk}")
            print(f"  ct.dual Akkd: {'PASS' if akkd_ok else 'FAIL'} {r_akkd}")
        except Exception as e:
            print(f"  ct.dual failed: {e}")

    # Simple pass/fail based on max diff
    aqk_pass = aqk_max_diff < 1e-2
    akkd_pass = akkd_max_diff < 1e-2
    status = "PASS" if (aqk_pass and akkd_pass) else "FAIL"
    detail = f"aqk_max={aqk_max_diff:.3e} akkd_max={akkd_max_diff:.3e}"
    return status, detail


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.environ.get("KDA_ISC_OUT_DIR", os.path.join(REPO_ROOT, "tests", "kda_intra_sub_chunk_dual", "output"))
    only = os.environ.get("KDA_ISC_CASE", "").strip()

    cases = CASES
    if only:
        cases = [c for c in cases if c.name == only]
        if not cases:
            print(f"Unknown case: {only}")
            sys.exit(1)

    print(f"device={device} out_dir={out_dir} cases={len(cases)} HAS_GPU={HAS_GPU} HAS_CT={HAS_CT}")

    results = []
    for case in cases:
        try:
            status, detail = run_case(case, device, out_dir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            status, detail = "ERROR", str(exc)
        results.append((case.name, status, detail))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, status, detail in results:
        print(f"  {status:8s} {name}: {detail}")

    if any(s in ("FAIL", "ERROR") for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
