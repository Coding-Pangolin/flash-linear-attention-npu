# P4 — Deeper soft-pipe sub-plan (ChunkKdaBwdWyDqkgFused)

> Parent: Cursor `wy_dqkg_next_opt` §5 / `wy_dqkg_epilog_iter` E4  
> Baseline after P0–P3: **~5.84 ms** (P1a Gate MTE2 PP). Stretch ≤ **0.8 ms**.  
> Status: **E4 trial done** — `USE_WIN_SOFT_LEAD_V2` implemented (PostS2→S0(next bank)→Stage3); suite green; board **+0.37 ms** → macro **0**, code kept.

## 1. Why P4

P1–P3 closed what they could:

| Knife | Result |
|-------|--------|
| P1a Gate MTE2∥V PP | **−0.06 ms**, default on |
| P1b Epilog MTE2 PP | reject (+0.30 ms) |
| P2 Mask/partial/BAR | suite green; med flat |
| P3a FIX∥MTE2 reopen | **model AICORE 507015** — keep 0 |
| P3b L0 dbuf | skipped (P3a not green) |

Still **~7×** above 0.8 ms. Remaining headroom is structural: deeper window soft-pipe / cross-bank overlap, not more local BAR/MTE2 tweaks.

## 2. Target steady-state (schematic)

```text
Prefill:
  Stage0Vec + Cube Stage0(+necessary Stage1 handshake) × PREFILL(=2)

Steady (per window w):
  Epilog/Store(w) overlaps Gate/Kg of next bank more deeply than today
  Hot path flags: only C_S* / V_* counting (0x2)
  slotFree: bookend only (Set×4 at Process start; Wait before Stage0(w+2) reuse)

Current I5 (keep as floor — do not regress):
  Prefill Stage0 × 2
  Post(w) → Stage0(w+2)   // WaitFree still before Cube Stage0(w+2)
```

## 3. Deadlock / safety checklist (must pass before coding)

Copy into the implementing PR description; tick before merge.

- [ ] **Flag semantics**: all CrossCore Set/Wait use raw `0x2` counting; no `0x1` on C_S*/V_* hot path.
- [ ] **Empty head**: odd HV still handshake (Set/Wait counts match even when headCnt=1).
- [ ] **Cube ↔ Vec mirror**: every Set on one side has Wait on the other with same order over (task, window, BK).
- [ ] **No I5b**: never schedule `Post(w+1)` before Cube `Stage0(w+2)` WaitFree — proven Vec stall.
- [ ] **No unsynced multi-DataCopy**: Gate/Epilog multi-issue only via AllocEventID PP (or serial Sync).
- [ ] **Bank reuse**: WaitFree(slot) before Cube writes Stage0 into a recycled slot; Prefill depth ≤ NUM_GM_SLOTS/heads.
- [ ] **Dump / hang**: if model hangs, dump flag timeline before reverting to lockstep; do not silently drop dual-AIV.
- [ ] **Suite + model**: short-T suite green **and** T8192 model msprof (no 507015 / hang) before default-on.
- [ ] **Single-variable**: one Process-schedule change per trial; Δ ≤ −0.05 ms to keep.

## 4. Candidate schedule moves (ordered)

1. **Post-tail ∥ next-bank Stage0Vec**  
   After Vec Epilog(w) Set V_MASK / barrier, allow Stage0Vec(w+prefill) to run while Cube still finishes Stage3(w) — only if flags already permit; audit Wait on C_S3.

2. **BK-inner Cube/Vec skew**  
   Within a window, start Gate BK0 of head1 while Cube still on Stage2 BK_last of head0 — higher risk; needs BK-level flag or per-head banks.

3. **Prefill=3** (if slots allow)  
   Only if NUM_GM_SLOTS and WaitFree math close; measure bubble vs register pressure.

4. **Algorithm / tile cut** (out of soft-pipe)  
   If after (1)–(3) still ≫ 2 ms: consider BK/BV retile or fusion split — separate plan.

## 5. Explicit non-goals (do not retry)

- I5b Post≻WaitFree  
- Epilog kPark reuse  
- Kg∥Gate interleave / VS0-once / EXP_GN_PARK  
- Gate unsynced triple DataCopy  
- FIX_MTE2 + L0_AB_DBUF until Vec-bound lifts enough that Cube bubble is measurable **and** Preload races are re-audited

## 6. Acceptance

| Milestone | Gate |
|-----------|------|
| Design PR | This file + checklist in PR body |
| First schedule trial | suite green + model no ECC/hang |
| Default-on | Δ ≤ −0.05 ms vs 5.84 ms baseline |
| Stretch | ≤ 0.8 ms (may need tile/algorithm work beyond soft-pipe) |

## 7. Suggested first patch (follow-up)

Touch only `Process` / `RunWindow*` in `*_cube.h` + `*_vector.h` with mirrored order; leave DirectTileGemm and Gate PP alone. Land behind `USE_WIN_SOFT_LEAD_V2=0` until measured.
