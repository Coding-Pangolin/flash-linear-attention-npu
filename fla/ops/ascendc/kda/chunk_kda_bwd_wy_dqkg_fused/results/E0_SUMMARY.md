# E0 Epilog-iter baseline (2026-07-29)

Shape: B1 H=HV=32 T8192 K128 V128 BT64 bf16, state_v_first=false

| Metric | Value |
|--------|-------|
| Task Duration | **5942.28 µs (~5.94 ms)** |
| aic_cube_ratio (med) | 0.042 |
| aic_mte2_ratio (med) | 0.301 |
| aic_fixpipe_ratio (med) | 0.113 |
| aic_scalar_ratio (med) | 0.428 |
| aiv_vec_ratio (med) | 0.320 |
| aiv_mte2_ratio (med) | 0.205 |
| aiv_mte3_ratio (med) | 0.143 |
| aiv_scalar_ratio (med) | 0.487 |
| aiv_scalar_vector_stall (med µs) | ~1164 |
| aiv_scalar_wait_id10 (med µs) | ~2898 (AIV Barrier) |

Bound: still **AIV / scalar / BAR**. Sim (existing `prof_sim_t1024_p1a`): BAR 30%, MOVEMASK 12%, MTE_UB_GM 19%. Next: E1 Mask/Select slim.
Prof: `results/prof_e0/OPPROF_*`
