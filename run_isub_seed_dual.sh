#!/usr/bin/env bash
# Repo-root one-click: seed-aligned NPU↔CPU dual (CPU RNG, no dump transfer).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test/run_intra_sub_chunk_seed_dual.sh" "$@"
