#!/usr/bin/env bash
# Source CANN + fla_npu vendor env for torch_custom tests.
#
# On a new machine, set explicitly before sourcing:
#   export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
#   export VENDOR_SET_ENV=/path/to/fla_npu_transformer/bin/set_env.bash
#   export CANN_OPP_LIB=/path/to/opp/vendors/fla_npu_transformer/op_api/lib
#
# Example (CANN 9.0, huangjunzhe layout — vendor 无 set_env.bash，只需 OPP lib):
#   export CANN_SET_ENV=/data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
#   export CANN_OPP_LIB=/data/huangjunzhe/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib
#   source torch_custom/fla_npu/test/setup_cann_env.sh
set -euo pipefail

if [[ -z "${CANN_SET_ENV:-}" ]]; then
  for candidate in \
    /data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh \
    /data/zs/run/8.5/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh
  do
    if [[ -f "$candidate" ]]; then
      CANN_SET_ENV="$candidate"
      break
    fi
  done
fi

if [[ -z "${CANN_SET_ENV:-}" || ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: CANN_SET_ENV not set and no default set_env.sh found." >&2
  echo "  export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${CANN_SET_ENV}"

if [[ -z "${VENDOR_SET_ENV:-}" ]]; then
  cann_root="$(cd "$(dirname "${CANN_SET_ENV}")/.." && pwd)"
  for candidate in \
    "${cann_root}/opp/vendors/fla_npu_transformer/bin/set_env.bash" \
    "${cann_root}/vendors/fla_npu_transformer/bin/set_env.bash" \
    /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash
  do
    if [[ -f "$candidate" ]]; then
      VENDOR_SET_ENV="$candidate"
      break
    fi
  done
fi

if [[ -n "${VENDOR_SET_ENV:-}" && -f "${VENDOR_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${VENDOR_SET_ENV}"
else
  echo "WARN: VENDOR_SET_ENV not found; custom ops may be missing." >&2
fi

if [[ -n "${CANN_OPP_LIB:-}" ]]; then
  export LD_LIBRARY_PATH="${CANN_OPP_LIB}:${LD_LIBRARY_PATH:-}"
elif [[ -n "${VENDOR_SET_ENV:-}" ]]; then
  opp_lib="$(cd "$(dirname "${VENDOR_SET_ENV}")/../op_api/lib" 2>/dev/null && pwd || true)"
  if [[ -n "$opp_lib" && -d "$opp_lib" ]]; then
    export LD_LIBRARY_PATH="${opp_lib}:${LD_LIBRARY_PATH:-}"
  fi
else
  # CANN 9.0 layout: vendor 可能无 set_env.bash，仅 op_api/lib
  ascend_root="$(cd "$(dirname "${CANN_SET_ENV}")/.." && pwd)"
  for opp_lib in \
    "${ascend_root}/../cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib" \
    "${ascend_root}/opp/vendors/fla_npu_transformer/op_api/lib"
  do
    if [[ -d "$opp_lib" ]]; then
      export LD_LIBRARY_PATH="${opp_lib}:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
