# F3a' — BK128 / owned-compact arena

> 日期：2026-07-29  
> 状态：**done** — `USE_BK128=1` default on；见 [`F3A_SUMMARY.md`](F3A_SUMMARY.md)

## 为何裸 `MAX_BK=128` 不可行

Gate/Epilog 按 **全 BT×BK 面板** 开 UB：

| 峰值 | BK=64 | BK=128 |
|------|-------|--------|
| Epilog `8*BT*BK` f32 | 128 KB | **256 KB** |
| + T arena `6*BT*BK` | 48 KB | **96 KB** |
| 合计 | ~180 KB OK | **≫192 KB FAIL** |

## 解锁路径（已落地）

双 AIV 实际只算 `nr≈BT/2` 行。arena 按 **owned 行** 开：

```text
ARENA_F32 = 8 * (BT/2) * BK = 8*32*128 = 32768 f32 = 128 KB
ARENA_T   = 6 * (BT/2) * BK = 48 KB
合计 ≈ 180 KB → GO
```

Kg/Gate/Epilog：`arena[r0*bk]` → 本地 compact `[0..nr)`；`CopyWs*` 仍带 `r0` 写回 WS。  
Epilog 在 state / MASK dA 复用 arena 前 park dg → `dkPartialWs`。
