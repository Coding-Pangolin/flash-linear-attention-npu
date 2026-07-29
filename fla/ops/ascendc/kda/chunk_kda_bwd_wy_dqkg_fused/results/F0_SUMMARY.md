# F0 Next-iter baseline (2026-07-29)

Shape: B1 H=HV=32 T8192 K128 V128 BT64 bf16, state_v_first=false  
Defaults: E1 stack (`USE_MASK_SELECT_SLIM=1`, V2/E2/E3 off)

| Metric | Value |
|--------|-------|
| Task Duration | **5911.34 µs (~5.91 ms)** |

Prof: `results/prof_f0/`  
Next: F1 `USE_MERGE_BARRIER_ONLY`
