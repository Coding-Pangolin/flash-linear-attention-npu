# P0 next-plan baseline (2026-07-29)

Shape: B1 H=HV=32 T8192 K128 V128 BT64 bf16, state_v_first=false

| Metric | Value |
|--------|-------|
| Task Duration | **5905.12 µs (5.91 ms)** |
| aic_cube_ratio (med) | 0.043 |
| aic_scalar_ratio (med) | 0.429 |
| aic_mte2_ratio (med) | 0.303 |
| aic_fixpipe_ratio (med) | 0.113 |
| aiv_vec_ratio (med) | 0.322 |
| aiv_mte2_ratio (med) | 0.202 |
| aiv_scalar_ratio (med) | 0.495 |
| aiv_scalar_vector_stall (med µs) | ~1202 |
| aiv_scalar_wait_id10 (med µs) | ~2976 (AIV Barrier) |

Bound: still Vec / scalar / barrier. Next: P1 MTE2∥V ping-pong.
Prof: `results/prof_p0_next/OPPROF_*`
