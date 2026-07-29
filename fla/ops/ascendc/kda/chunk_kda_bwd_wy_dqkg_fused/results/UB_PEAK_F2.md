# F2 UB peak table (2026-07-29)

AtlasA2 AIV UB budget ≈ **192 KB**. Current Init (vector.h):

| Buffer | Formula | BK64/BV64 | BK128/BV64 | BK64/BV128 |
|--------|---------|-----------|------------|------------|
| arenaF32 | `8*BT*BK` f32 | **128 KB** | **256 KB** | 128 KB |
| arenaT | `6*BT*BK` bf16 | **48 KB** | **96 KB** | 48 KB |
| beta+dbAcc | `2*BT` f32 | 0.5 KB | 0.5 KB | 0.5 KB |
| smallBuf | `4*BK` f32 | 1 KB | 2 KB | 1 KB |
| brcbBuf | `BT*8` f32 | 2 KB | 2 KB | 2 KB |
| mask+zero | ~0.5 KB | 0.5 KB | 0.5 KB | 0.5 KB |
| **合计** | | **~180 KB** OK | **~357 KB** **FAIL** | **~180 KB** OK |

## Verdict

| Config | Go/No-go | Note |
|--------|----------|------|
| **F3a `MAX_BK=128`** | **NO-GO** | Arena scales with BK; need redesign (spill / fewer live panels) before trial |
| **F3b `MAX_BV=128`** | **GO** | Arena sized by BK; V=128 → nBv=1; Stage0 peak `4*BT*BV` ≤ 8*BT*BK |

Next: F3b `USE_BV128` only (single-variable).
