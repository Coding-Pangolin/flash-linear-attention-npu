# ChunkKdaFwdPrepare（A2/A3）：UB / L1 / workspace 字节账与 4-head / 2-head 结论

> 常量来源：`op_kernel/chunk_kda_fwd_prepare_policy.h` 的 `Shape` / `Workspace` / `L1` / `Arch22Ub`
> 四个 namespace（下列数值由脚本直接从该头文件求值，非人工估算）。
> 结论：**4-head 调度窗口可行；UB 侧只能 2-head 驻留**。

## 1. 基础常量（A2/A3）

| 常量 | 字节 | 说明 |
|---|---:|---|
| `kHardwareBytes` | 192 KiB (`0x30000`) | A2/A3 UB 硬件容量 |
| `kUsableBytes` | **184 KiB** (`0x2E000`) | 可用（尾 8 KiB 保留） |
| `kPrivateBytes` | 72 KiB (`0x12000`) | 每个 AIV 的私有工作集 |
| `kSharedG` | 32 KiB | FP32 G（V0 产出 → V1/V3/V6 消费） |
| `kSharedScratch` | 8 KiB | Stage 内向量临时区 / beta gather offsets（**A2/A3 无 VF/RegBase**，只是普通向量指令的 scratch） |
| `L1::kCapacity` | 512 KiB | L1 容量 |
| `L1::kHeadLaneBytes` | 72 KiB | C2 payload（Qplus 16 + Kplus 16 + Kminus 40） |
| `L1::kPeak` | 368 KiB | L1 实测/布局峰值 |
| `Workspace::kSlotStride` | 105 KiB | 33 KiB context + 72 KiB payload |
| `Workspace::kArch22RawScoreBytes` | 20 KiB | C2 raw score relay（4 band：2+4+6+8 KiB） |
| `Workspace::kArch22SlotCount` | 4 | 每 workgroup 4 个 slot（对应 4 head） |
| `Shape::kHeadsPerGroup` | 4 | 调度窗口 |

## 2. UB 布局（184 KiB，正好用满）

| 区域 | 偏移 | 大小 | 生命周期内的用途 |
|---|---:|---:|---|
| private[0]（AIV0） | `0x00000` | 72 KiB | 按 stage 原地换义：<br>V0/V1：Q(0x0000,16K)、K(0x4000,16K)、GateOrKMinus(0x8000,…)、Kminus×4(0x8000/0x9000/0xB000/0xE000)、beta/state(0x10000…0x12000)<br>V3：Aqk/Lkk/Leaf0/Leaf1/B/X0/X1/NegX1/CompactRaw/AkkPack(0x0000…0x10000)<br>V6：Qg/Kg/VBeta/KBetaG(0x0000…0x10000) |
| 共享 G | `0x12000` | 32 KiB | `kSharedG` |
| 共享 scratch | `0x1A000` | 8 KiB | `kSharedScratch`、`kBetaGatherOffsets = kSharedScratch + 0x1F00` |
| private[1]（AIV1） | `0x1C000` | 72 KiB | 同 private[0] |
| 硬件保留 | `0x2E000` | 8 KiB | 不分配 |

**派生账**

| 方案 | 需求 | vs 184 KiB | 结论 |
|---|---|---:|---|
| 2-head 驻留（每 AIV 一个 head 工作集） | 72 + 40 + 72 = **184 KiB** | 0 | ✅ **采用（现状）** |
| 3-head 驻留 | 3×72 + 40 = 256 KiB | +72 | ❌ |
| 4-head 驻留 | 4×72 + 40 = **328 KiB** | +144 | ❌ **缺 144 KiB** |

> 每 head 的 72 KiB 是 V1 产出 `Qplus16 + Kplus16 + Kminus40` 的硬需求，无法压缩；
> 因此**UB 侧 4-head 同时驻留在 184 KiB 下不可能**，必须 2-head 驻留。

## 3. L1 布局（峰值 368 / 512 KiB）

| 区域 | 偏移 | 大小 | 用途 |
|---|---:|---:|---|
| 4 × head lane | `0x00000` | 288 KiB | C2 payload（每 head 72 KiB） |
| X0 | `0x48000` | 16 KiB | 每 head 4 KiB 槽位（C4 输入） |
| NegX1 | `0x4C000` | 16 KiB | 每 head 4 KiB 槽位（C5 输入） |
| T | `0x50000` | 16 KiB | 每 head 4 KiB quadrant（C4 输出→C5 输入） |
| Akk | `0x54000` | 32 KiB | 每 head stride 8 KiB（C7 的 A 操作数） |
| **峰值** | — | **368 KiB** | **余 144 KiB** |

| 窗口 | 需求 | vs 512 KiB | 结论 |
|---|---|---:|---|
| 4-head | 288 + 80 = 368 KiB | −144 | ✅ |
| 8-head | 576 + 80 = 656 KiB | +144 | ❌ |

## 4. workspace（GM）

| 项 | 值 |
|---|---|
| 每 slot | `0x1A400`(105 KiB) + `0x5000`(20 KiB relay) = **125 KiB** |
| 每 workgroup | 4 slot × 125 KiB = **500 KiB**（`kArch22SlotCount=4`） |
| 28 核合计 | ≈ 14 MB |
| 4-head / 2-head | 分别占 4 slot / 2 slot（GM 不是瓶颈；注意私有 workspace 偏移是 `uint32`，总量硬上限 4 GiB） |

## 5. 结论汇总

| 资源 | 4-head | 2-head | 采用 |
|---|---|---|---|
| 调度窗口（任务粒度） | ✅ | ✅ | **4-head**（尾波收益来自任务分配而非驻留） |
| L1 驻留 | ✅ 368/512 | ✅ 224/512 | **4-head**（余 144 KiB 可给 C7 RHS 预取） |
| workspace slot | ✅ 4 slot | ✅ 2 slot | **4 slot** |
| **UB 同时驻留** | ❌ 328 KiB > 184 | ✅ 184 KiB = 184 | **2-head** |

**⇒ A2/A3 的最终形态：4-head 调度窗口 + 2-head UB 驻留（pair 分两批）。**

**何时才降级到 2-head 窗口**：当某项优化要求"UB 内同时放 4 份 head 工作集"时
（例如 head 级流水、V6 与 V3 真并行），把 `kHeadsPerGroup` 改为 2：

- 代价：任务数 ×2（每任务固定开销——事件 alloc/release、prologue——摊薄变差）；
- 收益：L1 余量 144 → 288 KiB，UB 约束松开（每 AIV 仍 72 KiB）。
