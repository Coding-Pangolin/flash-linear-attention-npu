# E4 WIN_SOFT_LEAD_V2 trial (2026-07-29)

Schedule (P4 §4.1): Prefill=1; Cube `PostS2(w) → Stage0(w+1 other bank) → Stage3(w)`;
Vec `PostBody(w) → Stage0Vec(w+1) → PostTail(w)`. Mirrored flags; no I5b.

| Metric | Value |
|--------|-------|
| Task Duration | **6260.69 µs (~6.26 ms)** |
| vs E1 (~5.89) | **+0.37 ms** — reject |
| Suite | green |
| Macro | `USE_WIN_SOFT_LEAD_V2=0` (code retained under `#if`) |

Prof: `results/prof_e4_v2/`
