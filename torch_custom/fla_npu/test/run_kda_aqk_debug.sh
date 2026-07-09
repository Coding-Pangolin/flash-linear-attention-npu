#!/usr/bin/env bash
# KDA Aqk NaN debug: use huangjunzhe CANN + fla_npu_transformer vendor (not /usr/local).
set -euo pipefail

CANN_ROOT=/data/huangjunzhe/Ascend/cann-9.0.0
VENDOR_ROOT=${CANN_ROOT}/vendors/fla_npu_transformer
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TEST_DEVICE_ID=${TEST_DEVICE_ID:-1}
CONDA_ENV=${CONDA_ENV:-wnc}

source /data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

source "${CANN_ROOT}/set_env.sh"
if [[ -f "${VENDOR_ROOT}/bin/set_env.bash" ]]; then
  source "${VENDOR_ROOT}/bin/set_env.bash"
fi

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}"
export LD_LIBRARY_PATH="${VENDOR_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
# dump_cce: DumpTensor/printf from kernel prints to screen via RUNTIME ParsePrintToLog.

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[env] ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
echo "[env] device=${TEST_DEVICE_ID}"

python3 - <<'PY'
import os
import torch
import torch_npu  # noqa: F401
import fla_npu  # noqa: F401

from torch_custom.fla_npu.test.test_npu_chunk_kda import (
    MODEL_CASE,
    _make_model_fused_inputs,
    _assert_finite,
)

dev = int(os.environ.get("TEST_DEVICE_ID", "1"))
torch.npu.set_device(dev)
print("npu device_count:", torch.npu.device_count())

bundle = _make_model_fused_inputs(torch.device(f"npu:{dev}"))
q, k, v = bundle["q"], bundle["k"], bundle["v"]
cs = bundle["chunk_size"]
scale = bundle["scale"]
initial_state = bundle["initial_state"]

# Use CPU fp32 reference gk to reach chunk_kda_fwd (skip gate_cumsum mismatch).
from torch_custom.fla_npu.test.test_npu_chunk_kda import _kda_gate_cumsum_reference

gk = _kda_gate_cumsum_reference(
    bundle["g_raw"].detach().cpu(),
    cs,
    A_log=bundle["a_log"].detach().cpu(),
    dt_bias=bundle["dt_bias"].detach().cpu(),
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
).to(device=q.device)

beta = torch.sigmoid(bundle["beta_raw"].float()).to(dtype=q.dtype)

got = torch.ops.npu.npu_chunk_kda_fwd(
    q, k, v, gk, beta, scale, cs,
    initial_state=initial_state,
    output_final_state=True,
    return_intermediate=True,
)
torch.npu.synchronize()
o, final_state, g_out, aqk, akk = got[0], got[1], got[2], got[3], got[4]
print("o", f"finite {int(torch.isfinite(o).sum())}/{o.numel()}")
print("aqk", f"finite {int(torch.isfinite(aqk).sum())}/{aqk.numel()} nan={int(torch.isnan(aqk).sum())}")
print("aqk sample", aqk[0, 0, :4, :4].detach().cpu())
print("chunk_kda_fwd done — check stdout above for kernel DumpTensor/printf")
PY
