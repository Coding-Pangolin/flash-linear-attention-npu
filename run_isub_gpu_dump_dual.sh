#!/usr/bin/env bash
# Repo-root one-click wrapper for intra_sub_chunk GPU dump dual.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test/run_intra_sub_chunk_gpu_dump_dual.sh" "$@"
