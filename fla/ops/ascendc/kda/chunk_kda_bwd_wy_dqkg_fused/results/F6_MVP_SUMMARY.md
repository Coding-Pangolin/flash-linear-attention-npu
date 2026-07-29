# F6 MVP — stage split OpA/B/C（精度门禁）

> 日期：2026-07-29  
> 基线 fused Task Dur ≈ **3.86 ms**（F3a'）

## 落地

- tiling：`stageId` / `taskBegin` / `taskEnd` / `numSlots`（env：`FLA_WY_DQKG_*`）
- Kernel：`ProcessStageA/B/C`；stage 用 `numSlots=HV` 唯一 slot；关 MASK_SOFT_LEAD
- Python：`split_stages=True` + 共享 workspace；`n_stream` 分 task 区间

## 验收

| 项 | 结果 |
|----|------|
| suite fused | **green** |
| suite `split_stages=True` N=1 | **green** |
| model e2e fused | ~26.8 ms |
| model e2e split N=1 | ~101.7 ms（launch 税）|
| model e2e split N=2 | ~107.6 ms |
| default | **off**（fused 默认）|

## 结论

精度契约已通；N=1/2 端到端尚被 **多次 launch** 拖累，未过性能门禁。  
下一刀：增大 task batch（或升 Op Attr）、减 launch 次数 / 真占用重叠后再评 default。
