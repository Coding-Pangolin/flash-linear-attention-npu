# F3a' — owned-compact arena + `USE_BK128=1`

> 日期：2026-07-29  
> 基线：F3b `USE_BV128=1` → Task Dur ≈ **4741 µs (~4.74 ms)**

## 改动

- `USE_BK128=1` → `MAX_BK=128`（nBk=1）；`ARENA_BT_ROWS=MAX_BT/2`
- `USE_OWNED_ARENA`：Kg/Gate/Epilog 面板按 owned 半行开 UB（`8*(BT/2)*BK`）
- host tiling `MAX_BK=128` 同步 SlotLayout / WS
- Epilog：park dg→`dkPartialWs` 后再复用 arena 做 state / MASK dA（防 clobber）
- `CopyStridedOutOwned(..., compactSrc)` 区分 compact BK 与 full-panel Stage0/Mask

## 验收

| 项 | 结果 |
|----|------|
| suite | **all cases passed** |
| msprof Task Dur | **3861 µs (~3.86 ms)** |
| Δ vs F3b | **−0.88 ms**（门禁 −0.05）|
| 裁决 | **default on** |

Prof：`results/prof_f3a/OPPROF_20260729170331_LFXWWCACONHIHAPC`
