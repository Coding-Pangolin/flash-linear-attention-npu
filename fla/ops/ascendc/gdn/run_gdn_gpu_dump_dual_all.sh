#!/usr/bin/env bash
# Serial GPU dual benchmark for recompute_wu, fwd_h, bwd_dhu.
#
# Usage:
#   ./run_gdn_gpu_dump_dual_all.sh <DUMP_ROOT> [OPTIONS]
#
# Examples:
#   ./run_gdn_gpu_dump_dual_all.sh /data/GPU_DUMP
#   ./run_gdn_gpu_dump_dual_all.sh /data/GPU_DUMP --output-dir /data/gdn_dual_out
#   ./run_gdn_gpu_dump_dual_all.sh /data/GPU_DUMP --case phase_1_fix_1
#   ./run_gdn_gpu_dump_dual_all.sh /data/GPU_DUMP --phase prefix:phase_1_ -sc 100000
#   TEST_DEVICE_ID=2 ./run_gdn_gpu_dump_dual_all.sh /data/GPU_DUMP --device 2
#
# Output layout (per operator):
#   <output-dir>/
#     recompute_wu/
#       logs/recompute_wu.log
#       recompute_wu_gpu_dump_dual_report.json
#       viz/<case_name>/...
#     fwd_h/
#       logs/fwd_h.log
#       fwd_h_gpu_dump_dual_report.json
#       viz/<case_name>/...
#     bwd_dhu/
#       logs/bwd_dhu.log
#       bwd_dhu_gpu_dump_dual_report.json
#       viz/<case_name>/...
#     summary.json

set -euo pipefail

GDN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOMPUTE_DIR="${GDN_DIR}/chunk_gdn_fwd/recompute_wu_fwd/test"
FWD_H_DIR="${GDN_DIR}/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests"
BWD_DHU_DIR="${GDN_DIR}/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test"

DUMP_ROOT=""
OUTPUT_DIR=""
DEVICE_ID="${TEST_DEVICE_ID:-0}"
STOP_ON_FAIL=false
NO_VIZ=false
SAMPLE_COUNT=""
RECOMPUTE_PHASE="bwd"

CASE_ARG=""
CASES_ARG=""
PHASE_ARG=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_gdn_gpu_dump_dual_all.sh <DUMP_ROOT> [OPTIONS]

Run recompute_wu, fwd_h, bwd_dhu GPU dual benchmarks serially.
Logs, JSON reports, and ct.viz images are stored per operator.

Options:
  --output-dir DIR       Output root (default: <DUMP_ROOT>/gdn_gpu_dump_dual_out)
  --device N             NPU device id (default: TEST_DEVICE_ID or 0)
  --case NAME            Single case directory name under DUMP_ROOT
  --cases a,b,c          Comma-separated case names
  --phase PHASE          Case filter: all | prefix:phase_1_ | prefix:gva_
  -sc, --sample-count N   ct.viz sample count (default: 200000 in python)
  --no-viz               Skip ct.viz
  --recompute-phase P    recompute_wu dump phase: fwd | bwd | any (default: bwd)
  --stop-on-fail         Stop after first operator failure
  -h, --help             Show this help

Any unrecognized args are forwarded to all three python test scripts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE_ID="$2"
      shift 2
      ;;
    --case)
      CASE_ARG="$2"
      shift 2
      ;;
    --cases)
      CASES_ARG="$2"
      shift 2
      ;;
    --phase)
      PHASE_ARG="$2"
      shift 2
      ;;
    -sc|--sample-count)
      SAMPLE_COUNT="$2"
      shift 2
      ;;
    --no-viz)
      NO_VIZ=true
      shift
      ;;
    --recompute-phase)
      RECOMPUTE_PHASE="$2"
      shift 2
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=true
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$DUMP_ROOT" ]]; then
        DUMP_ROOT="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$DUMP_ROOT" ]]; then
  echo "[ERROR] DUMP_ROOT is required" >&2
  usage
  exit 2
fi

DUMP_ROOT="$(cd "$DUMP_ROOT" && pwd)"
if [[ ! -d "$DUMP_ROOT" ]]; then
  echo "[ERROR] dump root not found: $DUMP_ROOT" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${DUMP_ROOT}/gdn_gpu_dump_dual_out"
fi
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

export TEST_DEVICE_ID="$DEVICE_ID"

COMMON_ARGS=(--dump-root "$DUMP_ROOT")
[[ -n "$CASE_ARG" ]] && COMMON_ARGS+=(--case "$CASE_ARG")
[[ -n "$CASES_ARG" ]] && COMMON_ARGS+=(--cases "$CASES_ARG")
[[ -n "$PHASE_ARG" ]] && COMMON_ARGS+=(--phase "$PHASE_ARG")
[[ -n "$SAMPLE_COUNT" ]] && COMMON_ARGS+=(-sc "$SAMPLE_COUNT")
$NO_VIZ && COMMON_ARGS+=(--no-viz)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${EXTRA_ARGS[@]}")
fi

declare -A OP_RC=()
declare -A OP_REPORT=()
declare -A OP_LOG=()

run_one_op() {
  local op_name="$1"
  local py_script="$2"
  local op_out="${OUTPUT_DIR}/${op_name}"
  local log_dir="${op_out}/logs"
  local viz_dir="${op_out}/viz"
  local log_file="${log_dir}/${op_name}.log"
  local report_file="${op_out}/${op_name}_gpu_dump_dual_report.json"

  mkdir -p "$log_dir" "$viz_dir"

  local -a op_args=("${COMMON_ARGS[@]}")
  op_args+=(--viz-dir "$viz_dir" --report "$report_file")
  if [[ "$op_name" == "recompute_wu" ]]; then
    op_args+=(--dump-phase "$RECOMPUTE_PHASE")
  fi

  echo ""
  echo "================================================================"
  echo "  Operator: ${op_name}"
  echo "  Script:   ${py_script}"
  echo "  Log:      ${log_file}"
  echo "  Report:   ${report_file}"
  echo "  Viz:      ${viz_dir}/"
  echo "  Device:   ${TEST_DEVICE_ID}"
  echo "================================================================"

  set +e
  {
    echo "=== ${op_name} GPU dual benchmark ==="
    echo "started_at: $(date -Iseconds)"
    echo "dump_root:  ${DUMP_ROOT}"
    echo "command:    python3 ${py_script} ${op_args[*]}"
    echo ""
    python3 "$py_script" "${op_args[@]}"
    echo ""
    echo "exit_code: $?"
    echo "finished_at: $(date -Iseconds)"
  } 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  OP_RC["$op_name"]="$rc"
  OP_REPORT["$op_name"]="$report_file"
  OP_LOG["$op_name"]="$log_file"

  if [[ "$rc" -eq 0 ]]; then
    echo "[PASS] ${op_name}"
  else
    echo "[FAIL] ${op_name} (see ${log_file})" >&2
    if $STOP_ON_FAIL; then
      write_summary
      exit "$rc"
    fi
  fi
}

write_summary() {
  local summary_file="${OUTPUT_DIR}/summary.json"
  python3 - "$summary_file" "$DUMP_ROOT" "$OUTPUT_DIR" "$DEVICE_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
dump_root = sys.argv[2]
output_dir = Path(sys.argv[3])
device_id = sys.argv[4]

ops = [
    ("recompute_wu", "recompute_wu_gpu_dump_dual_report.json"),
    ("fwd_h", "fwd_h_gpu_dump_dual_report.json"),
    ("bwd_dhu", "bwd_dhu_gpu_dump_dual_report.json"),
]

payload = {
    "dump_root": dump_root,
    "output_dir": str(output_dir),
    "device_id": device_id,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "operators": {},
    "total_passed": 0,
    "total_failed": 0,
    "all_passed": True,
}

for op_name, report_name in ops:
    op_dir = output_dir / op_name
    report_path = op_dir / report_name
    log_path = op_dir / "logs" / f"{op_name}.log"
    viz_dir = op_dir / "viz"
    entry = {
        "status": "missing",
        "report": str(report_path),
        "log": str(log_path),
        "viz_dir": str(viz_dir),
    }
    if report_path.is_file():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        entry["status"] = "pass" if data.get("failed", 0) == 0 else "fail"
        entry["total"] = data.get("total", 0)
        entry["passed"] = data.get("passed", 0)
        entry["failed"] = data.get("failed", 0)
        entry["results"] = data.get("results", [])
        if entry["failed"]:
            payload["all_passed"] = False
            payload["total_failed"] += int(entry["failed"])
        payload["total_passed"] += int(entry.get("passed", 0))
    else:
        payload["all_passed"] = False
        payload["total_failed"] += 1
    payload["operators"][op_name] = entry

summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Summary -> {summary_path}")
if payload["all_passed"]:
    print("ALL OPERATORS PASSED")
else:
    print("SOME OPERATORS FAILED", file=sys.stderr)
PY
}

echo "GDN GPU dual benchmark (3 operators)"
echo "  dump_root:   ${DUMP_ROOT}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  device:      ${TEST_DEVICE_ID}"
echo "  recompute:   --dump-phase ${RECOMPUTE_PHASE}"

run_one_op "recompute_wu" "${RECOMPUTE_DIR}/test_recompute_wu_gpu_dump_dual.py"
run_one_op "fwd_h" "${FWD_H_DIR}/test_fwd_h_gpu_dump_dual.py"
run_one_op "bwd_dhu" "${BWD_DHU_DIR}/test_bwd_dhu_gpu_dump_dual.py"

write_summary

failed_ops=0
for op in recompute_wu fwd_h bwd_dhu; do
  [[ "${OP_RC[$op]:-1}" -ne 0 ]] && failed_ops=$((failed_ops + 1))
done

echo ""
echo "================================================================"
echo "  Finished: $((3 - failed_ops))/3 operators passed"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Summary:  ${OUTPUT_DIR}/summary.json"
echo "================================================================"

exit "$([[ $failed_ops -eq 0 ]] && echo 0 || echo 1)"
