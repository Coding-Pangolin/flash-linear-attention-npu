#!/usr/bin/env bash
# Shared logging helper for run_*_gpu_dump_dual.sh wrappers.

gpu_dump_dual_log_dir() {
  local dump_root="${1:-}"
  local script_dir="${2:-.}"
  if [[ -n "$dump_root" && -d "$dump_root" ]]; then
    echo "${dump_root}/logs"
  else
    echo "${script_dir}/logs"
  fi
}

gpu_dump_dual_run_python() {
  local py_script="$1"
  local log_dir="$2"
  local op_tag="$3"
  shift 3

  mkdir -p "$log_dir"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${log_dir}/${op_tag}_gpu_dump_dual_${ts}.log"
  local latest_link="${log_dir}/${op_tag}_gpu_dump_dual_latest.log"

  echo "[INFO] log: ${log_file}"
  set +e
  {
    echo "=== ${op_tag} GPU dual benchmark ==="
    echo "started_at: $(date -Iseconds)"
    echo "command:    python3 ${py_script} $*"
    echo ""
    python3 "$py_script" "$@"
    rc=$?
    echo ""
    echo "exit_code: ${rc}"
    echo "finished_at: $(date -Iseconds)"
    exit "${rc}"
  } 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  ln -sfn "$(basename "$log_file")" "$latest_link" 2>/dev/null || true
  return "$rc"
}
