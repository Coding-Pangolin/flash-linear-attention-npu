#!/usr/bin/env bash
# Serial CPU dual benchmark for recompute_wu, fwd_h, bwd_dhu from cases.json.
#
# Usage:
#   TEST_DEVICE_ID=2 ./run_gdn_cpu_dual_casesjson.sh --smoke
#   TEST_DEVICE_ID=2 ./run_gdn_cpu_dual_casesjson.sh --op recompute_wu
#   TEST_DEVICE_ID=2 ./run_gdn_cpu_dual_casesjson.sh --cases gva_fix_3,gva_var_2
#
set -euo pipefail

GDN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOMPUTE_SH="${GDN_DIR}/chunk_gdn_fwd/recompute_wu_fwd/test/run_recompute_wu_cpu_dual_casesjson.sh"
FWD_H_SH="${GDN_DIR}/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/run_fwd_h_cpu_dual_casesjson.sh"
BWD_DHU_SH="${GDN_DIR}/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/run_bwd_dhu_cpu_dual_casesjson.sh"

OP_FILTER=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_gdn_cpu_dual_casesjson.sh [OPTIONS] [-- extra py args]

Run recompute_wu / fwd_h / bwd_dhu CPU dual benchmarks serially.
Logs under fla/ops/ascendc/gdn/dual_benchmark_logs/<op>/cpu_dual_*/

Options:
  --op NAME         recompute_wu | fwd_h | bwd_dhu | all (default: all)
  --smoke           built-in small cases for quick validation
  --cases LIST      comma-separated cases.json names
  --no-viz          skip ct.viz
  -h, --help        show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op)
      OP_FILTER="$2"
      shift 2
      ;;
    --smoke)
      EXTRA_ARGS+=(--smoke)
      shift
      ;;
    --cases)
      EXTRA_ARGS+=(--cases "$2")
      shift 2
      ;;
    --no-viz)
      EXTRA_ARGS+=(--no-viz)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OP_FILTER="${OP_FILTER:-all}"
RC=0

run_op() {
  local name="$1"
  local script="$2"
  echo ""
  echo "========== GDN CPU dual: ${name} =========="
  if ! bash "${script}" "${EXTRA_ARGS[@]}"; then
    RC=1
  fi
}

case "${OP_FILTER}" in
  recompute_wu) run_op recompute_wu "${RECOMPUTE_SH}" ;;
  fwd_h)        run_op fwd_h "${FWD_H_SH}" ;;
  bwd_dhu)      run_op bwd_dhu "${BWD_DHU_SH}" ;;
  all)
    run_op recompute_wu "${RECOMPUTE_SH}"
    run_op fwd_h "${FWD_H_SH}"
    run_op bwd_dhu "${BWD_DHU_SH}"
    ;;
  *)
    echo "unknown --op ${OP_FILTER}" >&2
    exit 2
    ;;
esac

exit "${RC}"
