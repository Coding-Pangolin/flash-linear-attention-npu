# F6b — multi-task stage banks（减 launch）

> 日期：2026-07-29  
> 基线：F6 MVP e2e split N1 ~101.7 ms；fused ~20–27 ms

## 改动

- tiling：`numSlots = hv * ceil(rangeTasks / usedCore)`
- kernel：`taskBank_` + `SlotOf = bank*hv + window*2 + h`
- Python：默认 `batch_tasks = total_tasks`（一次 A→B→C）；`FLA_WY_DQKG_BATCH_TASKS` 可缩小

## 验收

| 项 | 结果 |
|----|------|
| suite fused + split | **green** |
| model e2e fused | **~20.8 ms** |
| model e2e split N=1 | **~41.1 ms**（MVP ~101 → **−60 ms**）|
| model e2e split N=2 | **~51.4 ms** |
| max\|dq split−fused\| | **0** |
| default | **off** |

## 结论

Launch 税已砍大半，但仍约 **2× fused**，未过 default 门禁。  
下一刀：真多 stream 占用重叠（更小 WS bank / pipeline A∥B 跨 partition），或回 fused 主路径抠 AIV sync。
