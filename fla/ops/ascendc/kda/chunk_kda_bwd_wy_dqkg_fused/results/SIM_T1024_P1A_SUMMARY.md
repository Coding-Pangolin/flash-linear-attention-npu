# Simulator T=1024 (current P1a defaults) — 2026-07-29

Shape: `B1 H=HV=2 T1024 K128 V128 BT64 bf16`, `state_v_first=false`, warmup=1  
Cmd: `msprof op simulator --kernel-name=ChunkKdaBwdWyDqkgFused --soc-version=Ascend910B3`  
Out: `results/prof_sim_t1024_p1a/OPPROF_20260729145402_AVJVSOXNKDSFZPJO`  
Log: `results/prof_sim_t1024_p1a_run.log`

| Metric | Value |
|--------|-------|
| Total tick | **1,392,321** |
| Model RUN TIME (host wall) | ~231 s |
| finite_dq | True |

## instr_exe (all cores summed)

**AIV (32 veccores)** — total cycles ~29.9M

| Share | Instr |
|------:|-------|
| 30.4% | BAR |
| 11.9% | MOVEMASK |
| 9.8% | MOV_UB_TO_OUT |
| 8.5% | WAIT_FLAG_DEVI |
| 8.1% | ST_XD_XN_IMM (scalar-ish) |
| 5.5% | VCADD |
| 4.4% | MOV_OUT_TO_UB |
| 1.5% | VMUL |

**AIC (16 cubecores)** — total cycles ~6.5M

| Share | Instr |
|------:|-------|
| 27.6% | WAIT_FLAG_DEV |
| 22.7% | BAR |
| 14.1% | MOV_OUT_TO_L1_MULTI_ND2NZ |
| 12.1% | FIX_L0C_TO_DST |
| 5.5% | LOAD_2D |
| 0.9% | MMAD |

## Read for next knives

- Still **AIV-bound** in sim (vec cycles ≫ cube); Cube spends more on **WAIT/BAR** than MMAD.
- Vec hot: **BAR + MOVEMASK + store/load + CrossCore WAIT** — aligns with board `aiv_scalar`/`wait_id10` story; P4 soft-pipe / mask path more relevant than more Gate MTE2.
- Cube: Fix+ND2NZ+Wait bubble still visible → P3 FIX∥MTE2 remains theoretically relevant but model ECC blocks default-on.
