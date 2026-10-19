# A4′（叶子逆外迁 cube）实测结论：**不成立**（2026-10-09，234 实测）

> 机器/口径：`192.168.9.234` / 8×910B3 / device 7 / 1800 MHz，perf case 63
> （B1 HK96 HV96 T8192，chunk 64），warmup=2 / repeat=10，`m_a2.sh` 的 event 计时。
> 所有 tag 都可以用 `TAG=<tag> CID=63 W=2 R=10 bash m_a2.sh` 复现。

## 1. 为什么要做这一轮

`design_a2a3_e2_stage_attribution.md` 的 E2 屏蔽实测给出：

```text
case 63（A1 之后）        总时长 12,184.7 µs
  V0+V1   4,892 µs  40.1%
  V3      3,075 µs  25.2%   ← 其中「两个 32×32 叶子逆」2,362 µs = 19.4%
  V6      2,002 µs  16.4%
  C2      1,988 µs  16.3%
  C4+C5+C7 2,246 µs 18.4%   ← AIC 侧合计 busy 只有 ~21%，cube 本体 264 µs
```

据此形成了 A4′ 的推理：**AIC 有 ~79% 余量、cube 只有 1.4% busy ⇒ 把 AIV 上最贵的
叶子逆（19.4%）外迁到 cube**。前置（AIV 产 L / AIC 读 L / 16×16 增广求逆在设备上
验证通过）都已经做完，本轮把它**铺满到 4 个 16×16 块**并接线测量，得到决定性的结论。

## 2. 本轮做了什么（补丁与 tag）

| 文件 | 作用 |
|---|---|
| `patch_probe4.py` + `probe4_cube.txt` / `probe4b_tiles.txt` / `probe4_sent.txt` | AIV 造 4 个增广 tile（A0/D0/A1/D1）→ cube 各做 4 次平方 → 6×16 哨兵比对 |
| `patch_probe4x.py` + `probe4x_tiles2.txt` / `probe4x_cubenone.txt` | 同一套 tile，cube 侧可切换「真空转」以分离两侧代价 |

4 个 tile 的输入数据：`A0=D00(0:16,0:16)`、`D0=D00(16:32,16:32)`、
`A1=D11(0:16,0:16)`、`D1=D11(16:32,16:32)`（都是 L 的对角 16×16 块，
其逆就是 X0/X1 的对应对角块）⇒ 哨兵可以不用 X0/X1 全阵就能判定对错。

## 3. 实测结果（同一批次，同一台机）

| tag | 内容 | event (µs) | Δ vs `leafpair` |
|---|---|---:|---:|
| `leafpair` | A1 块状 norm + 两叶共享屏障（**当前最优，无 tile、无 cube 求逆**） | **11,922.8** | — |
| `base_inv2` | leafpair + 1 个 tile + cube 做 1 次 16×16 逆（旧 probe） | 12,309.0 | **+386** |
| `probe4c` | 4 个 tile（2 批搬运）+ **cube 空转** | 12,413.5 | **+491** |
| `probe4d` | 4 个 tile + cube **4 次** 16×16 逆 | 14,438.9 | **+2,516** |
| `probe4b` | 同 4d，但 tile 一块一块搬（4 次阻塞等待） | 14,489.5 | +2,567 |

**代价分解**（本次直接量出来的）：

```text
AIV 侧：4 个增广 tile 的生成 + 16 KiB 搬运（每 chunk-head）
        = probe4c − leafpair = +491 µs
cube 侧：4 次 16×16 增广求逆
        = 16 次 MMAD + 16 次 fixpipe(落 GM) + 12 次 Nd2Nz(回 L1)
        = probe4d − probe4c = +2,025 µs     ← 约 3.95 µs/chunk-head（512 个/chunk-head/核）
```

## 4. 结论：A4′ 净收益为负

把 AIV 的两个 32×32 叶逆（E2 屏蔽实测 **2,362 µs**）整个外迁，
净变化 ≈ **2,362 − (491 + 2,025) = −154 µs（略亏）**；而 32 级方案除了这 4 个
16×16 逆，还要再做**跨块项**（`M = C@A⁻¹`、`X0[16:32,0:32] = [−D⁻¹C A⁻¹, D⁻¹]`）
与 X0/X1 的装配、以及让 `T=B@X0`/`q10=−X1@T`/Akk 打包改吃 cube 结果 ⇒ 只会更亏。

两条根因（都是本轮实测出来的，而不是推测）：

1. **C220 上 L0C 不能直写 L1**：fp32 只能 fixpipe 落 GM、再 Nd2Nz 搬回 L1。
   一次 16×16 逆 = 4 个平方步 ⇒ 每 chunk-head 16 次 MMAD + 16 次 fixpipe(GM)
   + 12 次 Nd2Nz(L1) 的**串行往返**，每步约 250~350 ns，累计约 2 µs/chunk-head。
   cube 的算力（264 µs / 1.4%）根本不是瓶颈，**瓶颈是这条 GM 往返链**。
2. **本算子的核间握手是逐回合 lockstep**（AIV `ready` → AIC 该回合算完 `free`
   → AIV 进入下一回合），AIC 的"空闲"不能用来吸收这些工作量：
   加在 C4 上的时间就是直接加在关键路径上。

⇒ **"AIC 有 79% 余量所以把 AIV 的贵段外迁"这条推理在本算子上不成立**：
余量 ≠ 可用并行度。**A2/A3 的 stage 边界不需要为 A4′ 改动，8 个 stage 维持不动。**

## 5. 顺带拿到的两条工程结论（已记档，避免重犯）

1. **AIV 里写"大块临时区"必须避开 pair1 的私有区**：
   `kPrivateBase[1] = 0x1C000`，而 `kSharedScratch = 0x1A000` 的**可用安全段只有
   0x1A000~0x1C000 这 8 KiB**（更靠上的地址会落进 pair1 的 UB 私有区）。
   `probe4a` 把 4 个 tile 放在 `kSharedScratch + 0x2000`，V3 一覆盖就把 pair1
   "已被 MTE3 发出、还没读完"的 `aqk` 冲掉 —— 现象是 **case 61 的 aqk 只有
   head2/3 出错（NaN + 1.4% 超差）**，而其它输出完全正常。
   改成只用 `0x1A000~0x1C000` 后立刻恢复：case 60/61/80 全 PASS、
   误差与基线**逐位一致**（4.883e-04 / 9.766e-04 / 4.883e-04）。
2. **tile 一块一块搬 vs 一次搬 2 块**：14,489.5 vs 14,438.9 µs（差 51 µs，噪声级）
   ⇒ 这点微调不值得再花 build；真正的代价在 cube 侧。

## 6. 下一步该往哪打（按"收益/改动面"排序）

| 优先级 | 方向 | 依据/上限 | 改动面 |
|---|---|---|---|
| **P1** | **AIV 内部压叶逆**：把"每步一次 `Gather`"改成"循环前一次性转置布局" | Gather 实测 **1,264 µs（10.6%）**，上限 ~1.0~1.2 ms | 只在 `StageV3`/`V3Vf` 内部，不动 stage 边界、不动跨核契约 |
| P2 | 摘掉 AIC 的 C2/C4/C7 关键路径（跨 chunk-head 多缓冲 + 新 flag 配平） | AIC 侧合计 35% 都在关键路径上 | 大：round 结构、workspace 多缓冲、flag 全重排 |
| P3 | V1 三族指数与 V0 的 gate 指数去重 + V6 本体 | 1,460 µs / 2,002 µs | 中：需要先切一层确认结构 |
| — | A4′（叶子逆外迁 cube） | **本轮实测否定**（净 −0.15 ms，且复杂度高） | 不建议继续 |

## 7. 产物清单（234 `/home/admin123/kda_a2/`）

```text
patches : patch_probe4.py / patch_probe4x.py
          probe4_cube.txt / probe4b_tiles.txt / probe4x_tiles2.txt
          probe4x_cubenone.txt / probe4_sent.txt
tags    : probe4（4 tile + cube 逆，含 v3dump 前身） / probe4b（1 块 1 次搬）
          probe4c（4 tile，cube 空转） / probe4d（4 tile + cube 4 逆）
          base_inv2（= 本轮 baseline：1 tile + 1 逆） / leafpair（无 tile，最优）
logs    : build_probe4*.log / acc/probe4b/acc_*.log / acc/*/acc_61.log
```
