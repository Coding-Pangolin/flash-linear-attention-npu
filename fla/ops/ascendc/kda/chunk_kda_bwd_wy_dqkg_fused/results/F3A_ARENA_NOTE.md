# F3a' — BK128 / owned-compact arena（未落地）

> 日期：2026-07-29  
> 状态：**blocked → 设计备忘**；本轮不改代码

## 为何裸 `MAX_BK=128` 不可行

Gate/Epilog 按 **全 BT×BK 面板** 开 UB：

| 峰值 | BK=64 | BK=128 |
|------|-------|--------|
| Epilog `8*BT*BK` f32 | 128 KB | **256 KB** |
| + T arena `6*BT*BK` | 48 KB | **96 KB** |
| 合计 | ~180 KB OK | **≫192 KB FAIL** |

## 解锁路径（F3a'）

双 AIV 实际只算 `nr≈BT/2` 行。若 arena 按 **owned 行** 开：

```text
ARENA_F32 = 8 * (BT/2) * BK = 8*32*128 = 32768 f32 = 128 KB
ARENA_T   = 6 * (BT/2) * BK = 48 KB
合计 ≈ 180 KB → GO
```

**必须改**：Kg/Gate/Epilog 全部 `arena[r0*bk]` 全面板索引 → 本地 `owned[0..nr)`，`CopyWs*` 仍带 `r0` 写回 WS。

工作量：vector.h 约 Gate+Epilog+Kg 三段，单变量宏 `USE_BK128` + host `MAX_BK`/`SLOT` 同步；风险精度/偏航。

**建议**：放在 F6 切分之后，或作为 F6 OpB 内单独 retile（切分后单 op UB 更松）。

## 本轮裁决

不试裸 BK128；记 **F3a' owned-compact** 为后续刀，不挡 F6。
