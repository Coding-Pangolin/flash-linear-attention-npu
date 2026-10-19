# AIV 侧杠杆实测（2026-10-09，234）

> 机器/口径：`192.168.9.234` / 8×910B3 / device 7 / 1800 MHz，perf case 63
> （B1 HK96 HV96 T8192，chunk 64），`m_a2.sh` W=2/R=10；精度用 `acc_batch.sh`。
> 前置结论见 `kda_prepare_a2a3_aic_hidden_evidence.md`（AIC 侧被完全掩蔽）
> 与 `kda_prepare_a2a3_A4_prime_probe_result.md`（叶子逆外迁 cube 净亏）。

## 0. 结论速览

| 项 | 状态 | 证据 |
|---|---|---|
| `clean_lp` 基线（去掉 A4′ 脚手架） | **11,931.7 µs**，case 60 PASS | §1 |
| P1：叶逆"两列一批"合并 `Gather` | **否决**（会翻倍元素数） | §2（probe6） |
| 叶逆"在 AIV 内换更便宜的广播" | 结构性下限，暂无便宜路 | §3 |
| `V1`/`V6` 链上的"流水排空/延迟暴露" | **≈1,957 µs（16.4%）**，方向是块间软流水 | §5（probe7） |
| **`V0` 的 norm 标量压紧 + cumsum 串行链** | **−533 µs 已落地**（−416 / −109） | ✅ §9（P3，已验证精度） |
| 剩余杠杆：`V1` 两个 exp 族 / `V6` 两个 exp 族 | 待逐族定价 | §4 |

## 1. `clean_lp`：把已被否定的 A4′ 脚手架从源码里摘掉

A4′ 被实测否决后，它在源码里留下的三样东西都不应该继续挂着：
① `StageV3` 里的 `L00/L11` 影子写（8 KiB/chunk-head）、② relay 里的增广 tile 写、
③ `StageC4Inv16` 与 `StageC4` 里的 L00/L11 读入、`InitBuffer` 多要的 0x6000 字节。
重建方式（可复现）：`vec.h = vec_normblk4.h + patch_leafpair.py`，`cube.h`/`policy.h` 回 `git HEAD`。

| tag | 内容 | case 63 event (µs) | case 60 精度 |
|---|---|---:|---|
| `leafpair`（历史最优，同源） | A1 + 两叶共享屏障 | 11,912.2 | PASS 4.883e-04 |
| **`clean_lp`（本轮）** | 同上，且源码无 A4′ 残留 | **11,931.7** | PASS 4.883e-04 |
| `base_inv2`（改前的工作树） | leafpair + A4′ 脚手架（1 tile + 1 次 cube 逆） | 12,309.0 | PASS |

⇒ **去掉脚手架回收 ~380 µs**，工作树回到干净的最优状态（源码里已无 `probe4/probe5/E2B/StageC4Inv16` 残留）。

## 2. probe6：叶逆的 `Gather` 是"按元素"计费 ⇒ P1 否决

### 2.1 探针设计（**不改变正确性**）

`SolveTwoLeavesRightLooking` 每步一次 `Gather`，每步只取 `rowsA` 行（`rowsA` 从 31 递减到 1），
但每行要写满 32 lane（= 把同一个标量放大 32 倍）。计划中的 P1 是"两列一批"：
**Gather 条数减半、每条读的元素翻倍**（静态偏移表要求每列占满 32 行）。

于是先做一个不改正确性的探针：把每次 Gather 的行数直接写满 32 行
（多读的行不会被后面的 `Mul/Sub` 消费，输出逐位不变）——
这样"条数不变、元素数翻倍"，就能判定 Gather 是按条数还是按元素计费。

### 2.2 数据

| tag | Gather 行数 | 条数/chunk-head | case 63 event (µs) | case 60 精度 |
|---|---|---:|---:|---|
| `clean_lp` | `rowsA`（均值 16） | 62 | **11,931.7** | PASS 4.883e-04 |
| `probe6` | 固定 32 | 62（不变） | **12,991.5（+1,060）** | PASS 4.883e-04（逐位一致） |

### 2.3 判读

- 条数不变、元素翻倍 ⇒ **+1,060 µs** ⇒ **Gather 的开销随"元素数"走，几乎与条数无关**；
- 反推当前那 62 次 Gather 的总代价 ≈ **1.0~1.3 ms** ✓ 与 E2 里"只关掉每步 Gather"的
  1,264 µs 相互印证（这一项是**可信的**，因为它屏蔽的是同一 stage 内的一条向量指令）；
- ⇒ **P1（两列一批）必然亏**：它把元素数翻倍（+1.06 ms 量级），只省条数（≈0）✗。

## 3. 为什么叶逆在 AIV 内也没有"便宜的广播"

每步的数学是 `X[k+1:,:] −= X[k,:] ⊗ L[k+1:,k]`（rank-1 更新）：需要一个"每个标量复制 32 lane"
的系数块，而 AscendC 的向量指令里：

- `Mul` 的 src 广播粒度是 **1 个 32 B block（8 个 fp32 lane）**，且广播是"整块对所有 repeat"，
  没法表达"每个 repeat 一个不同标量、且该标量在 32 lane 内复制"；
- `Brcb` 每个源元素只写 **1 个 block（8 lane）**，复制 4 次需要 4 条指令、且落点是错位的；
- 换算成"紧凑列向量 + 乘法"需要把行/列角色互换（转置 X 布局），但最后 Akk 打包是按行 cast，
  转置会再引入一次转置代价。
⇒ **32× 广播是当前外积形式的固有代价**，`Gather` 已经是能表达它的最省指令（1 条/步）。
因此叶逆在 AIV 内的继续压缩空间只剩"mul/sub/屏障"，而不是 Gather。

## 4. 剩下的杠杆（下一步逐族定价）

叶逆、AIC 侧、A4′ 都已被排除后，AIV 上还剩两处**元素吞吐 × 算子链长**的大头：

```text
V1（两个 exp 族，合计 ≈ 28.7K 元素/chunk-head，每个元素 ~9 条向量指令）
   Kminus 族：SubRowMinusTile → Maxs → Mins → Muls → Exp → Cast(bf16→fp32) → Mul → Cast(bf16)  （8 行一块）
   Qplus/Kplus 族：SubTileMinusRow → Maxs → Mins → Muls → Exp → Cast → Mul → Cast → Cast → Mul → Cast
V6（两个 exp 族 + V_beta/K_beta_g，≈ 24.6K 元素/chunk-head）
```

两者都是"每个元素 6~10 条向量指令"，且都有 8 行分块 + 每块 ~10 个 `PipeBarrier<PIPE_V>`；
屏蔽实测给的上界分别是 **1,460 µs** 与 **2,002 µs**。定价方式（每个变体 1 次 build）：

1. **逐族屏蔽**：分别只掐掉 `V1` 的两个 exp 族、`V6` 的两个 exp 族，看 AIV `vec` 与总时长各掉多少；
2. **屏障定价**：把同一 stage 内、同一 V 流水的冗余 `PipeBarrier<PIPE_V>` 去掉（V 流水本身顺序执行），
   看是否真有收益（叶逆那次的 93 个屏障只值 266 µs，说明单条屏障 ≈5.6 ns，需实测确认）；
3. 若 `V1` 的某一族确实贵，再评估"用一族 exp + 倒数/标量因子"替换的可行性（注意 clamp 域不同 ⇒ 会影响饱和区行为，必须过精度）。

## 5. probe7 / probe7b：V1/V6 的 `PipeBarrier<PIPE_V>` **不能删**（删了精度炸）

### 5.1 动机与做法

`V1Vf` / `V6Vf` 的每个 7 行小块都是"SubRowMinusTile/SubTileMinusRow → Maxs → Mins →
[Muls] → Exp → Cast → Mul → Cast"这样一条链，链内每一步之间都插了 `PipeBarrier<PIPE_V>()`
（两个函数静态共 49 条；动态每 chunk-head 约 200+ 条：`V1` 两个循环 × ~10 块 × ~10 条）。
直觉是"V 流水本来就顺序执行，同流水内的屏障可以删"，于是做了两版：

| tag | 做法 | case 63 event (µs) | 精度 |
|---|---|---:|---|
| `clean_lp` | 基线 | **11,931.7** | PASS 4.883e-04 |
| **`probe7`** | `V1Vf`/`V6Vf` 的 49 条全删 | **9,974.6（−1,957，−16.4%）** | **FAIL：worst=3.436e+11** |
| **`probe7b`** | 只保留 5 条（紧跟 `Exp`/两个 Sub helper 之后），其余 44 条删除 | （未测时长） | **FAIL：worst=3.436e+11（同一失败值）** |

### 5.2 判读（这条比收益更重要）

- **这些屏障是"必须"的**：本硬件/CANN 版本上，同一条 V 流水里的 RAW **不能**靠硬件顺序保证
  （probe7 直接炸到 3.4e11，probe7b 只留 5 条同样炸）⇒ **同流水屏障不是"保险"，是语义的一部分**；
- 那 **−1,957 µs（16.4%）** 说明：`V1`/`V6` 的**真正开销是这条链上的"流水排空/延迟暴露"**，
  而不是算子本身的元素吞吐。每个屏障都强迫前面的长延迟算子（尤其 `Exp`）把 latency 露出来，
  不能与后面的独立工作重叠；
- ⇒ **正确方向不是删屏障，而是"块间软流水"**：用两组缓冲让 block i 的 `Mul/Cast` 与
  block i−1 的 `Exp` 互相重叠（屏障只保护真正的 RAW），把暴露的 latency 藏起来。
  这是 A1（块状 norm）同一类手法在 V1/V6 上的推广，收益上限就是这 ~2 ms（16%），
  改动面是 `V1Vf`/`V6Vf` 两个函数的链式重排（不动 stage 边界、不动 flag）。

## 6. P2（把 V0 的输出契约从 `G` 改成 `E(G)`）—— **实测+推导否决**

### 6.1 原本的想法

文档一直把"`V1` 的 exp 族与 `V0` 重复计算（上限 1,460 µs）"列为 P2：
既然 `E(G−Gref) = E(G)·E(−Gref)`，那就在 `V0` 里把 `E(G)` 算一次（存 32 KiB 共享区，
替掉 `G`），`V1` 的两个族退化成"1 次乘/除 + 1 次 clamp"，`V6` 同理。

### 6.2 为什么这条路**不成立**（数学 + 实测）

`G` 的存储域是 **log2 指数**，clamp 到 `[-126, 120]`（`ExpDomain::kV1Bf16*Base2`）——
这个边界就是 **fp32 的指数范围**（`2^-126` 是最小正规数）。也就是说：
**当 `|G| > 126` 时，`E(G)` 在 fp32 里下溢/上溢，值本身就不再可表示**；
而 `V1` 真正需要的是**差值** `Gref_s − G[row]`，它在 `|G|` 很大时**仍然在范围内**
（例：`Gref=-160, G=-320 ⇒ 差值=160`，clamp 到 120 后仍有意义）。
一旦只存 `E(G)`，两者都变成 `2^-126`，比值恒为 1 ⇒ **差值信息被抹掉**，结果完全错。

**实测（本机现有 precision case 的 `gk` 输出 = 累积 `G`）**：

| case | 形状 | max abs(G) | `|G|>126` 的元素占比 |
|---|---|---:|---:|
| **60** | B1 H32 T160 | **257.5** | **36.97%** |
| **61** | B6 H6 T1084 | **248.0** | **46.12%** |
| 13 / 26 / 68 / 80 | 小 shape | 0.44 / 0.83 / 7.62 / 2.07 | 0% |

⇒ **模型规模的 case（60/61）里，三分之一到近一半的元素都越过 ±126** ⇒
`E(G)` 单存是**结构性错误**，不是精度损失问题。**P2 按原形态关闭。**

（顺带解释了设计里 clamp 的位置为什么必须在"存储域"：它是 fp32 指数边界的安全网，
而不是 bf16 的边界——bf16 的指数范围是 ±128，值域上是 `2^±128`，两者不能混用。）

## 7. 所有已知杠杆的最终状态

| 杠杆 | 上限 | 状态 |
|---|---:|---|
| `V0`/`V1` 的 clamp+exp+cast 链（A1 块状 norm） | 已落地 −6.6 ms | ✅ 已做 |
| **V6 的 kg 族**（`E(clamp(g_last-g))`） | **556 µs（实测，probe8）** | ❌ 结构性必需（与 §6 同一反例）⇒ 只能靠块间软流水重叠 |
| **V1/V6 链上的"流水排空 / 延迟暴露"** | **−1,957 µs（16.4%，删屏障的上界）** | ⏳ **唯一还有量级的点**：只能靠块间软流水把 latency 藏起来（屏障删不得，§5） |
| A4′：叶逆外迁 cube | +2,516 vs 收益 2,362 | ❌ 实测否决（`A4_prime_probe_result.md`） |
| AIC 侧调度/搬移（回合 3→2、V6 提前、外迁） | ≈0 或负 | ❌ 实测否决（`aic_hidden_evidence.md`） |
| P1：叶逆 `Gather` 两列一批 | +1,060（元素翻倍） | ❌ 探针否决（本文 §2） |
| V1/V6 删 `PipeBarrier<PIPE_V>` | −1,957 但精度炸 | ❌ 屏障是语义（本文 §5） |
| P2：V0 契约改 `E(G)`、V1/V6 去 exp | — | ❌ 结构性错误（本文 §6） |
| **`V6` 自己的两个 exp 族**（输入是**原始 gate**、量级小、未饱和） | 2,002 µs（上界） | ⏳ 唯一还没被证否的 exp 去重点；需先逐族定价 |
| `V0` norm 尾部的"标量压紧"循环（64 次 GetValue/SetValue + 一次同步往返） | **416 µs** | ✅ **P3-A 已落地**（§9） |
| `V0` cumsum 的 63 步串行链（每步 1 个屏障） | **109 µs**（屏蔽法给的 333 µs 偏高） | ✅ **P3-B 已落地**（log 扫描，§9；改变 `gk` 末位） |
| `V0` norm 尾部剩下的"逐行 Muls 标量"循环 | ≈0.3~0.4 ms（估） | ⏳ 现有原语表达不了（per-row 广播粒度只到 8 lane） |
| `V3` 叶逆的 mul/sub/屏障 | ~1.1 ms | ❌ 结构性下限（本文 §3） |

⇒ **AIV 侧还剩三个可做的点（V6 的 exp 去重、norm 的标量循环、cumsum 的分块扫描），
合计上限 ≈0.9~1.0 ms（8%）**；再往下就是"换算法/换架构"的问题了。

## 8. probe8：给 V6 的 kg 族定价，并说明它也是结构性的

V6Vf 每个 7 行块里有三族：`qg/qBeta` 族（`E(clamp(g))`）、`kg` 族
（`E(clamp(g[last]-g))`）、`qgScaled` 族。把 **kg 族整段屏蔽**（只做屏蔽，不追求正确）
测得的代价：

| tag | 内容 | case 63 event (µs) | Δ |
|---|---|---:|---:|
| `clean_lp` | 基线 | 11,931.7 | — |
| **`probe8`** | 屏蔽 V6 的 kg 族（sub + clamp×2 + muls + Exp + cast + mul + cast） | **11,375.8** | **−556** |

**这 556 µs 也不能靠"E(G) 相除"省掉**：V6 的 `g` 是 `gk` 输出（**累积 G**，本机 case
60/61 上 |G| 达 257/248、37%/46% 越过 ±126），而 kg 族的 clamp 是 ±80 ——
`E(g[last])/E(G[r])` 在 G 越界时同样会抹掉差值信息（与 §6 同一个反例）。
⇒ kg 族的 sub+clamp+exp 链是**结构性必需**，只剩"把它与其它块重叠"这一条路
（= 与 §5 同一个方向：块间软流水）。

## 9. **P3：两个 AIV 小改已落地（−533 µs，−4.5%）** incl 精度验证

两个改动都在 `V0Vf`，互相独立，一起上（补丁 `patch_probe9.py`，可 `--revert`）：

| 变体 | 内容 | case 63 event (µs) | Δ |
|---|---|---:|---:|
| `clean_lp` | 改前的最优 | 11,931.7 | — |
| **`clean_lp2`** | A + B 全上 | **11,398.9（复测 11,406.2）** | **−533（−4.5%）** |
| `probe9a` | 只上 A | 11,515.4 | −416 |

**A（−416 µs）：去掉 L2 norm 尾部那次"标量压紧"**。
`WholeReduceSum` 的结果在 dst 上**连续落位**（A1 文档 + 实测），所以
`for row: rowSum.SetValue(row, sumBlocks.GetValue(row))` 这 64 次标量读写
（外加一次 `V_S`/`S_V` 往返）是多余的 —— 直接原地做 `Adds(eps)→Sqrt→Div` 即可。
这一条也顺带证明：norm 里剩下的开销主要在**标量↔向量往返**，不在向量吞吐。

**B（−109 µs）：token 维 cumsum 从 63 步串行换成 log 扫描**。
原来 `for row: Add(g[row], g[row], g[row-1]) + PipeBarrier` = 63 次流水排空
（E2 屏蔽实测 333 µs，实际值 109 µs ⇒ 屏蔽法确实高估）；
现在 Hillis-Steele：`log2(validRows)` 轮、每轮 2 条块指令 + 1 个屏障，
在 `g`（共享区）与 `tileBuf`（私有区 0x8000，本 Stage 的 norm/gate 已用完）之间乒乓。
⚠ 求和顺序变成树形 ⇒ **`gk` 的末位与改动前不同**（相对 ~1e-7）。

**精度（5 个用例，全部 PASS，且 worst 值与改动前逐位相同）**：

```text
case 60: 4.883e-04    case 61: 9.766e-04    case 80: 4.883e-04
case 13: 1.192e-07    case 26: 2.384e-07    （全部 matched=1.000000）
```

⇒ 树形求和的舍入差异没有改变任何用例的最坏误差（误差由别处主导）。

## 10. **P4：norm 的逐行缩放改成 Brcb + 整块 Mul（−540 µs，−4.7%）**

`P3-A` 的实测说明 norm 尾部的开销主要在**标量↔向量往返**；P4 把剩下那趟也消掉：

```text
旧：for row: scales[row] = rstdOut.GetValue(row)        // 64 次标量读 + 一次 V_S/S_V
    for row: Muls(staging_row, staging_row, scales[row], 128)   // 64 条逐行指令
    for row: Cast(src_row, staging_row, CAST_RINT, 128)         // 64 条
新：Brcb(brcbScalars, rstdOut, 8, {1,8})                // 每行 1 个 32B block
    ReplicateRowScalarsToRepeat(...) + MulTileByRowScalars(...) // 本仓 V6 beta 路径同款 helper
```

数学不变（同一对 fp32 操作数、同一个乘法）⇒ 结果应与改动前一致。

（⚠ 过程教训：P4 那版我一开始把源码快照搞错了——`vec_best.h` 其实是"只上了 P3-A"的状态，
于是 `clean_lp3` 实际是 **A+P4**、不是 A+B+P4。已修正并重建，下表是校正后的口径；
**以后快照必须紧跟"刚构建通过"的那份源码**。）

| tag | 内容 | case 63 event (µs) | Δ |
|---|---|---:|---:|
| `clean_lp` | A1 + 两叶共享屏障（P3 之前） | 11,931.7 | — |
| `probe9a` | + P3-A（norm 标量压紧） | 11,515.4 | −416 |
| `clean_lp2` | + P3-A + P3-B（cumsum log 扫描） | 11,398.9 / 11,388.2（W2R10 / W4R20） | −533 |
| `clean_lp3` | + P3-A + P4（缺 P3-B 的中间态） | 10,866.1 / 10,848.1 | −1,066 |
| **`clean_lp4`** | **+ P3-A + P3-B + P4（当前最优）** | **10,797.1（W4R20）** | **−1,135（−9.5%）** |

⇒ **P4 在 A+B 基础上再省 ≈600 µs**（11,388~11,399 → 10,797）；
相对原始基线 18,848.7 µs 累计 **−42.7%**。

**精度：case 60/61/80/13/26/68 全 PASS**（worst 4.883e-04 / 9.766e-04 / 4.883e-04 /
1.192e-07 / 2.384e-07 / 4.883e-04，matched 全 1.000000）。

> **方法学更正（本轮发现）**：`acc_ab.py` 的读数**不是逐次确定的**——同一个包（`clean_lp2`）
> 连跑两次 case 68，`out 0` 的 max_abs 会分别给出 `1.907e-06` 与 `4.768e-07`、
> `out 1` 给出 `2.441e-04` 与 `0.000e+00`。所以**"与改动前逐位一致"这种判据不可靠**，
> 精度比较应当看"是否 PASS + 量级是否同一档"（本轮的差异都在 1 个 bf16 ULP 内）。

## 11. 下一步（已找到机制）：用 `HardEvent::V_V` 做"块间软流水"

`probe7` 说明：**删屏障**不行（精度炸），那 1,957 µs 的本质是"每个屏障 = 一次 V 流水排空
⇒ 每个长延迟算子（尤其 `Exp`）的 latency 都被摊在外面"。要把它藏起来，需要
"**只等依赖项、不让无关工作一起等**"的机制。查 CANN 9.1 头文件确认它存在：

```text
/usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/impl/basic_api/kernel_event.h
  enum class HardEvent  { ..., V_V, ... };          // 公共枚举里有 V_V
  enum class HardEventAiv { MTE2_V, V_MTE2, MTE3_V, V_MTE3, V_V, ... };  // AIV 专用枚举里也有
（A2/A3 = __NPU_ARCH__ 2201，走 HardEventAiv 这一支 ⇒ V_V 可用 ✓）
```

**用法**：`SetFlag<HardEvent::V_V>(id)` 标记"到此为止的 V 工作"，把**无关的独立工作**
发在 `SetFlag` 与 `WaitFlag` 之间 ⇒ `WaitFlag` 只等被标记的那一批 ⇒ 独立工作与它重叠。
这正是 `PipeBarrier<PIPE_V>`（排空一切）做不到的事。

**改造方案（V1Vf 先做）**：

1. `kTileRows` 7 → **5**（640 fp32/缓冲 = 2.5 KiB），把 `factor` 双缓冲、`stage` 仍单份
   ⇒ 3×640 = 1920 fp32 = 7.5 KiB ✓ 正好塞进共享 scratch 可用的 7936 B；
2. 每个块的计算链不变（算子与顺序不变 ⇒ 舍入顺序不变），但 `SetFlag<V_V>` 之后
   立刻发 **下一块**的独立部分（`Sub*` 用的 helper / `Cast` 读 kHat·qHat 等），
   `WaitFlag<V_V>` 只放在真正消费上一块结果的地方；
3. 判据照旧：case 60/61/80 精度 + case 63 性能；预期把这 1,957 µs 的一部分收回来
   （先按 1/3~1/2 估算 ≈ 0.6~1.0 ms）。

## 12. 产物

```text
源码 : src_snap/vec_best2.h（= **clean_lp4** 的 vec.h，含 P3-A/P3-B/P4）；
       src_snap/vec_ab.h（P3-A+B）、src_snap/vec_A_only.h（只 P3-A）、
       src_snap/vec_clean_lp.h（P3 之前）
补丁 : patch_probe6.py（gather 行数探针）、patch_probe7.py（删全部同流水屏障）、
       patch_probe7b.py（只保留 5 条）、patch_probe8.py（屏蔽 V6 kg 族）、
       patch_probe9.py（P3-A/P3-B，支持 AB|A 与 --revert）、patch_probe10.py（P4）
tag  : **clean_lp4（当前最优，10,797.1 µs，−42.7% vs 原始 18,848.7）** /
       clean_lp3（A+P4）/ clean_lp2（A+B）/ clean_lp / probe6 / probe7 / probe7b /
       probe8 / probe9a
精度 : acc/clean_lp/acc_60.log、acc/probe6/acc_60.log（PASS，逐位一致）
        acc/probe7/acc_60.log、acc/probe7b/acc_60.log（FAIL，3.436e+11）
        acc/probe9/acc_{60,61,80,13,26}.log（全 PASS，worst 与基线逐位相同）
        acc/probe10/acc_{60,61,80,13,26,68}.log（全 PASS）
脚本 : check_g_range.py（读 precision case 的 gk 输出，量 |G| 与越界比例；§6 的数据来源）
```
