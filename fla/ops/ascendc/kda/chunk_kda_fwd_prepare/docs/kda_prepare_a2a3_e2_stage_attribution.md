# ChunkKdaFwdPrepare（A2/A3）分 Stage 屏蔽差分（E2）实测

> ## ⚠ 2026-10-09 更正：本表的差值**不能当成本分解用**
>
> §3 及之后那张"逐 Stage 单独屏蔽"的表，**所有项相加 = 14,203 µs > 总时长 12,184.7 µs**，
> 本身就不是一份可加的分解。2026-10-09 又做了一次**直接对照**（只屏蔽计算、保留全部
> 核间 flag 回合）：把 AIC 的 C2/C4/C5/C7 **全部**空转，总时长 11,912 → **12,285 µs
> （+370 µs，不降反升）**；单独空转 C2 也是 +20 µs。整核画像显示
> **AIV 跨度 10.17 ms ≈ kernel 总时长 11.88 ms**，AIC 的 cube 只有 264 µs
> ⇒ **AIC 侧完全被 AIV 掩蔽**。
>
> 也就是说：单 stage 屏蔽的差值只能当"上界提示"（因为屏蔽一个 stage 同时拿掉了它
> 内部的 wait/set 与指令发射压力），**不能用它论证"AIC 有 79% 余量 ⇒ 可以做 AIV→AIC 外迁"**
> —— A4′ 的实测否定（+2,516 µs > 收益 2,362 µs）与此一致。
> 详见 `kda_prepare_a2a3_aic_hidden_evidence.md` 与
> `kda_prepare_a2a3_A4_prime_probe_result.md`。

> 机器：**192.168.9.234**，8×910B3，容器 `kda_a2`（CANN 9.2.0-beta.1，py3.10+torch_npu2.12），
> 采集设备 device 7，频率 1800 MHz。时间：2026-10-08。
> 用例：ATK perf **case 63** = `B1 HK96 HV96 T8192`（BNSD，chunk 64，forward，全开关 True）。
> 口径：`torch.npu.Event`，warmup=1 / repeat=3（用例本身 19 ms，重复 3 次足够；基线两次独立构建差 0.2%）。

## 1. 方法：在调用点上屏蔽 stage（不是往函数体里插 return）

第一版做法是在 stage 函数体首行插 `return;`，**整核挂死**（AICore 100% 无输出）。原因：
stage 函数体内的 `WaitFlag/SetFlag`（`ioFree_`/`sharedFree_`/`mToMte1_`/`tReady_`/`fixToMte1_`）
被一起跳过，令牌链断裂。实际观测：上一轮 `e2_blank.sh` 的"恢复"步骤把已经打过补丁的文件
又拷回了 `.e2bak`，于是最后一次构建出来的算子同时带 `V0` 与 `C2` 两个 blank ⇒ 必挂。

第二版（本轮）只在 **`Process()` 的调用点**做手术（`e2_apply.py`）：

```cpp
/*E2B*/ if (false) { StageV0(chunk, valueHead, localHead, pair); }
```

- 被屏蔽 stage **内部**的 wait/set 成对消失 ⇒ 令牌链不断；
- `Process()` 里的 `CrossCoreWaitFlag/SetCrossCoreFlag` **一律保留** ⇒ 核间回合数不变；
- 唯一例外：`C7` 的 free 许可是在**函数体内**发的，屏蔽时必须在调用点补发
  `CrossCoreSetFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair])`。

`B_V0` 不能单独做：`V1` 等的是 `V0` 发的 `mte3ToV_`，所以必须 `V0+V1` 一起。
同理 `C5` 等 `C4` 的 `tReady_`、`C7` 等 `C5` 的 `fixToMte1_`。

## 2. 结果

基线 = 18,949.7 µs/call（同机独立重建两次：18,915.4 / 18,949.7）。

| 变体 | 屏蔽内容 | 时长 (µs) | **该 stage 的关键路径贡献** | 占总时长 |
|---|---|---:|---:|---:|
| `e2v2_base` | — | **18,949.7** | — | — |
| `B_V01` | `V0+V1`（norm+beta+gate+cumsum；Gref/Q±/K±） | **7,574.7** | **11,375** | **60.0%** |
| `B_V3` | `V3`（mask+beta+叶子逆+跨叶修正） | 16,096.0 | 2,854 | 15.1% |
| `B_V6` | `V6`（qg/kg/K_beta_g/V_beta） | 17,076.9 | 1,873 | 9.9% |
| `B_C2` | `C2`（4 band MMAD） | 17,092.8 | 1,857 | 9.8% |
| `B_C45C7` | `C4+C5+C7`（T / q10 / W,U 全部 MMAD） | **16,883.9** | **2,066**（三者合计） | **10.9%** |
| `B_C7` | `C7` 单独屏蔽 | — | *挂死，未取到*（见 §5.2） | — |

**五项贡献之和 = 11,375 + 2,854 + 1,873 + 1,857 + 2,066 = 20,025 µs > 基线 18,950 µs**，
说明这些 stage 之间有重叠（尤其是 `V3/V6/C2` 与 `C45C7` 的重叠），
所以**不能用"贡献之和"当账本**；能直接用的是"去掉谁省多少"。
另外 `B_V01` 之后只剩 7,574.7 µs，而 `V3+V6+C2+C45C7` 四项之和是 8,650 µs ——
也印证了同一件事：AIV 内部各 stage 互相部分隐藏。

## 3. 结论：A2/A3 上这个算子是"V0/V1 单点吃满 AIV"

1. **`V0+V1` 一项就占 60%** —— 这不是调度问题，也不是 cube 问题，是 AIV 上的
   **逐行标量/短向量循环**问题（见 §4 的代码证据）。任何"重排 stage / 省跨核回合 /
   优化 cube"的方案，天花板都只有剩下 40%。
   **进一步：`V0+V1` 的 11.4 ms 里有 8.6 ms 只是 Q/K 的 L2 norm**（见 §2.2 的属性消融）。
2. 后三项（`V3` 15.1% / `V6` 9.9% / `C2` 9.8%）量级相当，合计 35%；
   它们都已是最小必要的数学步骤，单独动任一项收益 ≤ 1.9 ms。
3. 与 E1 整核画像自洽：AIV vec 12.6 ms / AIV scalar 4.7 ms / AIC busy 3.4 ms。
   `V0+V1` 的 11.4 ms 基本就是 **AIV vec + AIV scalar 的主要来源**。

## 2.2 属性消融：`Q/K L2 norm` 一项 = 8.62 ms（45.5%）

算子本身带了运行时开关（tiling 里决定走哪个分支），所以**不用重编**就能把 `V0`
内部的几块分开定价。做法：把 case 63 复制成 63001…63006，只翻一个属性
（`make_ablate_json.py`），输入仍按原 shape 生成 —— **只用于计时，数值语义不成立**。

| 消融 | 翻了什么 | 时长 (µs) | 相对基线 |
|---|---|---:|---:|
| 基线 | — | 18,949.7 | — |
| `no_l2norm` | `use_qk_l2norm_in_kernel=False`：`V0` 跳过 **Q/K 逐行 L2 归一化** | **10,329.6** | **−8,620（−45.5%）** |
| `unsafe_gate` | `safe_gate=False`：gate 走非 safe 分支 | 19,764.4 | **+815（更慢）** |
| `no_exp2` | `use_exp2=False`：指数改 e 底 | 18,504.3 | −445 |
| `no_neg_eigval` | `allow_neg_eigval=False` | 18,904.7 | −45（噪声级） |
| `no_gate` / `no_beta_sigmoid` | `use_gate_in_kernel=False` / `use_beta_sigmoid_in_kernel=False` | — | tiling 直接拒绝（`161002`）：这两个开关要求 `g`/`beta` 以别的形态进来，本用例的 shape 不合法，**测不了** |

由此得到 `V0+V1` 的内部拆分：

```text
V0 + V1                      11,375 µs   (E2 调用点屏蔽)
  ├─ V0 的 Q/K L2 norm        8,620 µs   (45.5% of total)
  └─ V1 + V0 其余(gate/beta/cumsum)  ≈ 2,755 µs
```

**这条是整个 A2/A3 性能问题的主线**：一个"每行一条 Cast/Mul/ReduceSum + 一次
`GetValue/SetValue` 标量往返"的循环，占了近一半的算子时间。

### 2.3 同一个消融在 GVA 形状上复现（norm 占比 46~47%）

| 用例 | shape | 基线 | 关掉 L2 norm | norm 占比 |
|---|---|---:|---:|---:|
| 63 | B1 HK96 **HV96** T8192（HK=HV） | 18,949.7 | 10,329.6 | 45.5% |
| **56** | B2 HK16 **HV32** T11264（GVA，HV/HK=2） | 16,183.4 | 8,755.2 | **45.9%** |
| **65** | B1 HK**8** HV**24** T32768（GVA，HV/HK=3） | 17,572.1 | 9,339.7 | **46.8%** |

三个形状的 (chunk × value-head) 数都是 ~12k，norm 时长都是 8.2~8.6 ms ⇒
**norm 的成本只跟 `HV × chunk` 走，完全不看 `HK`**。

而 `StageV0` 里其实已经有"QK 输出归属"的概念（`IsQkOutputOwner(tiling, valueHead)`，
`vec.h:340`）：GVA 时同一个 key head 被 `HV/HK` 个 value head 重复用到，
只有 owner head 会把 `q_hat/k_hat` 写到 GM，**但 norm 的计算每个 value head 都重算了一遍**。

⇒ **case 65 这类 GVA（HV/HK=3）场景，"norm 按 HK 去重"的理论收益 ≈ 45% × (1−HK/HV) ≈ 31%**。

## 4. 为什么 `V0+V1` 这么大：代码级的直接证据

`op_kernel/arch22/chunk_kda_fwd_prepare_vec.h`：

### 4.1 `V0Vf` 的 Q/K L2 norm 是"每行一条指令 + 每行一次标量往返"

```cpp
for (uint32_t row = 0; row < validRows; ++row) {
    AscendC::Cast(scratch, q[row * kHeadDim], CAST_NONE, kHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(scratch[kHeadDim], scratch, scratch, kHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<...>(betaEff, scratch[kHeadDim], ..., reduceShape, true);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(betaEff, betaEff, epsilon, 1);
    AscendC::Sqrt(betaEff, betaEff, 1);
    AscendC::Duplicate(qOne, 1.0F, 1);
    AscendC::Div(betaEff, qOne, betaEff, 1);
    SetFlag/WaitFlag<V_S>;  const float qScale = betaEff.GetValue(0);   // ← 标量往返
    qRstd.SetValue(row, qScale);
    SetFlag/WaitFlag<S_V>;
    AscendC::Muls(scratch, scratch, qScale, kHeadDim);
    AscendC::Cast(q[row * kHeadDim], scratch, CAST_RINT, kHeadDim);
    ... 同样的 12 条对 k 再做一遍 ...
}
```

即：**每个 (chunk, head) 处理 64 行 × 2 个张量**，每行 ≈ 12 条向量指令 + 8 个 `PipeBarrier`
+ 1 次 `V_S/S_V` 硬件 flag 往返。按 case 63 的 12,288 个 (chunk,head) 折算，
仅 norm 一项就是数量级 **10⁶ 条指令**，而每条指令的有效宽度只有 128 个 fp32
（≈2 拍），绝大部分时间花在**指令发射、屏障和标量往返**上，不是算力上。

### 4.2 `V1Vf` 已经是块状，但它要重算一遍 `exp`

`V1` 按 7 行一块下发（`kTileRows = 7`，受共享 scratch 前 7 KiB 限制），
`clamp → Muls(exp 底换算) → Exp → Cast → Mul` 全块指令，比 `V0` 高效；
但 `Qplus/Kplus/Kminus` 三套都要按 `E(G−Gref)`/`E(Gref−G)` 重新计算，
与 `V0` 里已经算过的 gate 指数因子**重复**。

## 5. 已知的坑（给后续 agent）

1. **`return;` 式 blank 会挂死**：stage 体内的 flag 是令牌链的一部分。见 §1。
2. **`B_C7` 单独屏蔽会挂死，但 `B_C45C7`（一起屏蔽）正常**：说明挂死不是"free 许可
   没发"（补发语句是对的，`B_C45C7` 用的就是同一句），而是**单独去掉 C7 之后，
   AIV 侧提前拿到 free 与 C7 的 RHS 搬运/L1 head lane 复用产生了新的次序问题**。
   这条只影响"怎么测"，不影响结论：cube 侧三个 stage 合起来的贡献已经由
   `B_C45C7` 给了上界 2,066 µs。**要单独给 `C4`/`C5`/`C7` 定价，应该改用
   "保留结构、只把 MMAD 的 M/N/K 换成 16 或用空 MMAD"的做法，不要再整段屏蔽。**
3. 缓存输入：`kda_a2_bench.py` 第 5 个参数给 `.pt` 路径，第二次起 2~10 s 载入，
   否则每个变体要等 20 s~2.5 min 的 `torch.randn`（本机 CPU 被其他容器压满）。
4. `build.sh` 每次会重建 `third_party`（abseil/protobuf），**一个变体 ≈ 5 min**；
   7 个变体 ≈ 40 min，别并发跑（并发会得到 `CMAKE_CXX_COMPILER not set` 的假错误）。

## 6. 下一步（按收益排序）

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| **A1** | **`V0` 的 norm 改成块状**：`Cast` 整块 → `Mul` → 行内 `WholeReduceSum`/2D `ReduceSum` → `rsqrt`(64 元素) → 广播 → `Muls` → `Cast`（`AscendC::Brcb` 已确认存在于本机 CANN：`asc/include/basic_api/kernel_operator_vec_brcb_intf.h`，语义是"取 8 个元素各广播成一个 32B block"） | §2.2：norm 单独就是 8.62 ms / 45.5%；§4.1：现在 64×2 行 × ~20 条指令 ≈ 2,500 条/chunk-head | 指令数按数据量下限折算后可降到 1/3~1/2，**8.6 ms → 3~4 ms** |
| **A1'** | **GVA 下 norm 按 HK 去重**（`HV/HK = 3` 时理论省 2/3） | §2.3：norm 成本只随 `HV×chunk` 走；代码里已有 `IsQkOutputOwner` 但只省了 GM 写、没省计算 | case 65 上限 ~31% |
| **A2** | **gate 因子复用**：`V0` 算 `G`/`E(G)` 时把 `E(G−Gref)` 类因子留在 UB，`V1` 不再重算 | §4.2 | 省掉 `V1` 的三套 `Exp` |
| **A3** | **消掉每行 `V_S/S_V` 往返**：`beta`/`rstd` 用块状 `Brcb` 广播而不是 `GetValue/SetValue` | `aiv_scalar_stall_by_ub = 5.8 ms`、`aiv_scalar_vector_stall = 6.2 ms` | 直接打 AIV scalar 4.7 ms |
| A4 | 把叶子逆（32×32）从 `V3` 挪到 cube（Neumann 倍增，5 次 32³ MMAD，AIC 有 79% 余量） | E1：cube 只用 1.4% | `V3` 2.85 ms 的一部分 |
| A5 | `V6` 提前 + 省回合（原 D1） | E2：`V6` 9.9%，且只依赖 `V0` | 只在 AIV 降下来之后才有意义 |

**判据**：A1~A3 都是"同一 stage 内换实现"，位级必须与现状一致（ATK 抽 dense/GVA/varlen/尾块 ≥2 条）；
`A4` 涉及边界与同步，按 CANNBot 03 走。

---

## 7. A1 落地实验：把 `V0` 的逐行 L2 norm 改成块状（2026-10-08 晚，234 实测）

### 7.1 改了什么

文件：`op_kernel/arch22/chunk_kda_fwd_prepare_vec.h`（新增 `L2NormRowsBlock()`，
`V0Vf` 里的 64×2 行循环换成对 q/k 各一次块调用）。生成/回退脚本：`patch_norm.py`
（`vec.h.orig` 备份），新增代码在 `norm_block_helper.txt`。

```text
旧：for row in 0..63:  Cast->Mul->ReduceSum->Adds->Sqrt->Dup->Div
                       + V_S/S_V 标量往返 + Muls->Cast      （每行 ~20 条指令 + ~8 个屏障）
新：Cast 整块 -> Mul 整块 -> 折半 Add + WholeReduceSum 出 64 个行和
    -> 标量压紧一次 -> Adds/Sqrt/Dup/Div（块指令）-> 标量读一次
    -> 逐行 Muls/Cast（无逐行屏障）
```

折半 + `WholeReduceSum` 的写法照抄本仓 arch22 已上量的
`recurrent_gated_delta_rule.h::ReduceSumAddFoldK128`。staging（64×128 FP32 = 32 KiB）
在 KDA（FP32 gate）用 V1 之前空闲的 `kGateOrKMinus` 私有区，在 GDN（BF16 gate，
gate 自己就在该私有区）改用尚未写入的共享 G 区。

### 7.2 性能（perf case 63 = B1 HK96 HV96 T8192，同机同频，warmup 2 / repeat 10）

| 版本 | 时长 | 相对基线 |
|---|---:|---:|
| 基线（未改） | **18,832.0 µs** | — |
| 块状 norm | **12,066 ~ 12,098 µs** | **−35.9%** |

改动前基线在同一轮里量到 18,850 / 18,915 / 18,950 µs，改后两次独立构建量到
12,066 / 12,098 µs ⇒ 收益 **6.7 ms**，与属性消融给出的 norm 上限 8.62 ms 同量级。

### 7.3 精度（对 executor 自带 FP64 标杆，直接调用、不走 ATK）

ATK 的 accuracy 通道在本容器跑不通（baseline 也一样：
`KeyError: 'mixed_tolerance_bm'`、`npu_*_output_bytes=None`），所以用
`acc_ab.py`（复用同一 `executor` 的 `_run_npu/_reference`）做 A/B；`acc_cmp.py` 比两份 NPU 出参。

| 用例 | 形态 | 基线 max_abs | 块状 norm max_abs |
|---|---|---:|---:|
| 13 | BNSD T=33（单 chunk 33 行）, g **BF16** | 1.192e-07 | **1.192e-07** ✓ |
| 80 | TND T=17（块 8/9 行）, mode=save | 3.576e-07 | 4.883e-04 ✓ |
| 26 | BNSD T=65（尾块 **1**）, g **BF16** | 2.384e-07 | **2.384e-07** ✓ |
| 68 | BNSD T=65（尾块 **1**）, g **BF16** | 9.766e-04 | **9.766e-04** ✓ |
| 60 | BNSD T=160（尾块 32）, HK=HV=32 | 4.883e-04 | **4.883e-04** ✓ |
| 61 | BNSD B6 T=1084, HK=HV=6 | 9.766e-04 | **9.766e-04** ✓ |
| 62 | BNSD B1 T=1084, HK=HV=12 | 4.883e-04 | **4.883e-04** ✓ |

**结论：7/7 全过**，其中 6 条与基线**逐位同误差**，`case 80` 是 4.883e-04（bf16 1 ULP 量级，
远小于 atol=1e-2；它走的是下面的小尾块回退路径）。KDA（FP32 gate）与 GDN（BF16 gate）
两条分支、dense/TND/多 chunk/尾块 32/60 行都覆盖到了。

**分档结论**（这轮实测把边界钉死了）：

```text
validRows >= 32 : 走块状路径（case 13 的 33 行、case 60 的尾块 32 行都过）
validRows <  32 : 退回改动前的逐行实现（case 26/68 的尾块 1 行、case 80 的 8/9 行走这里）
```

`case 26`/`case 13` 这两条把"到底是 GDN 分支坏还是小尾块坏"分开了：
`case 13`（GDN、33 行、单 chunk）**通过** ⇒ GDN 分支本身没问题，问题只在小尾块。

再补一轮更宽的 l2norm 用例（baseline 与 normblk4 各跑一遍对比）：

| 用例 | 形态 | 基线 | normblk4 |
|---|---|---:|---:|
| 55 | T=65, exp2 + 小 epsilon, mode=none | 3.576e-07 | **3.576e-07** ✓ |
| 67 | BNSD T=1×12head, g BF16, mode=recompute | 4.768e-07 | **4.768e-07** ✓ |
| 69 | TND T=63 HK=1 HV=4, mode=recompute | 2.441e-04 | **2.441e-04** ✓ |
| 188 | BNSD T=65, mode=none | 2.441e-04 | 9.766e-04 ✓ |
| 71 / 74 / 75 / 198 | 变长 NTD/TND（cu_seqlens 由框架自动生成） | RUN FAIL | RUN FAIL（**两边一致**，是 harness/用例问题，不是回归） |

⇒ 累计 **11 条通过**，其中 9 条与基线**逐位同误差**；两边都失败的 4 条与本次改动无关。

### 7.4 期间踩到/修掉的两个坑（不是"讲道理"，都是实测逼出来的）

1. **`WholeReduceSum` 的结果在 dst 上是连续的**（`dst[repeat]`），不是"每个 repeat 占一个
   32B block"：按 block 步长去读时，第 i 行会读到第 8i 行的和，于是 `q_rstd` 大面积错、
   越过有效行后读到未初始化内存 ⇒ KDA 大 shape 出现 NaN。改成 `sumBlocks.GetValue(row)`
   后，KDA 分支全部与基线同误差。这条是从下面这组数据反解出来的（case 80）：
   `patched_sum(row)=0.288916/0.265016/0` 恰好等于 `true_sum(row 8/16/24)`。
2. **只设 `ASCEND_CUSTOM_OPP_PATH` 不足以切换版本**：op store 还会跟着
   `opp/vendors/fla_npu_transformer` 这个软链走，软链不改就会"两个版本量出同一个数"。
   `m_a2.sh` / `acc_batch.sh` 现在会同时改软链（这条曾经让我得到过一组假的 A/B）。

### 7.5 小尾块（validRows=1）那处：已用"分档 + 回退"收口，但根因未明

实测现象（`case 26`/`case 68` 的尾块，validRows=1）：块状路径给出的 `q/k` 恰好等于
**未归一化的原始值**（patched/base = 1/rstd），而 **rstd 本身是正确的**
（`q_rstd` 输出与基线逐位相同）。即"块算 rstd 正确、但缩放的 scale 被当成 1.0"。
同一份块状代码在 8/9、17、32、33、60、64 行上都正确 ⇒ 只有 `rows=1` 这一档异常，
根因尚未定位（候选：`rows=1` 时折半 Add / WholeReduceSum 的 repeat 语义）。

处理：块状路径加了准入门槛 `kBlockNormMinRows = 32`，`validRows < 32` 的尾块**走改动前的
逐行实现**（同一条 `L2NormRowsBlock` 内的回退分支，公式与改前一致）。尾块本身最多一个
chunk，回退不影响实测收益（12,190 µs，与不回退的 12,066 µs 同级）。

回退分支还踩到一个硬约束：**向量指令的 dst 必须 32B 对齐**。第一版回退把
`ReduceSum/Sqrt/Div` 的 dst 写成 `rstdOut[row]`（只有 4B 对齐）⇒ `case 80` 直接
`aicore exception 507015`。改成"在 32B 对齐的临时区里算完 scale，再
`SetValue` 写回 `rstdOut[row]`"后正常（与改动前的写法一致）。

### 7.6 复现方式

```bash
# 源码：应用/回退补丁
python patch_norm.py <vec.h>            # 打补丁（自动留 .orig）
python patch_norm.py <vec.h> --revert   # 回退
# 构建 + 装到独立 vendor（不覆盖别人的包）
TAG=normblk4 bash build_a2.sh
# 性能：perf case 63
JSON=.../atk_chunk_kda_fwd_prepare_perf_a2.json CACHE=.../case63.pt \
  TAG=normblk4 CID=63 W=2 R=10 bash m_a2.sh
# 精度：对 FP64 标杆
TAG=normblk4 bash acc_batch.sh 13 26 60 61 62 68 80
```

（当前推荐的候选包是 **`normblk4`**；`normblk2` 保留了"未加门槛"的版本，
`normblk3` 是最早一版回退，用来对照。）

---

## 8. A1 落地之后的重新归因（E2 v3，2026-10-08 深夜）

第一项改动落地之后，**旧的 stage 画像已经作废**（V0+V1 从 60% 掉到 40%），所以用同一套
屏蔽差分（`e2_v3.sh`，base = `vec_normblk4.h`）在**新基线 12,184.7 µs** 上重跑了一遍，
并把 V3 再往下切了一层（`e2_sub.sh` + `e2_sub.py`）。

### 8.1 新的分 Stage 画像

| 屏蔽对象 | 时长 (µs) | 贡献 (µs) | 占比 | 改前占比 |
|---|---:|---:|---:|---:|
| 新基线 | 12,184.7 | — | — | — |
| `V0+V1` | 7,292.6 | **4,892** | **40.1%** | 60.0% |
| `V3` | 9,109.9 | **3,075** | **25.2%** | 15.1% |
| `V6` | 10,182.3 | 2,002 | 16.4% | 9.9% |
| `C2` | 10,196.6 | 1,988 | 16.3% | 9.8% |
| `C4+C5+C7` | 9,938.8 | 2,246 | 18.4% | 10.9% |

（各项之和 14,203 µs > 12,185 µs，仍说明 stage 之间有重叠；只能看"去掉谁省多少"。）
**`V0+V1` 的 11,375 µs 里被消掉了 6,483 µs**，与总时长的下降（6.7 ms）一致。
AIC 侧四项加起来 4,234 µs（占比 ~35%），仍然不是第一矛盾。

### 8.2 V3 再切一层：叶子逆是 V3 的绝对大头

把 `V3Vf` 里两次 `SolveLeafRightLooking`（两个 32×32 叶子逆）单独包进 `if (false)`：

| 配置 | 时长 | 差值 |
|---|---:|---:|
| 新基线 | 12,184.7 | — |
| 屏蔽两个叶子逆 | **9,822.6** | **2,362 µs = 19.4% 整算子 / 77% of V3** |

⇒ **32×32 三角逆（2 个/chunk-head，各 32 步串行）是现在最清晰的"结构性错配"**：
它是短向量、强串行的活，正好压在 AIV 上；而 AIC 只有 21% busy、cube 本体 1.4%。
V3 剩下的 ~713 µs 才是 mask/beta/unpack 那部分。

### 8.3 更新后的重设计优先级（按实测收益/风险）

| # | 项 | 依据 | 上限 |
|---|---|---:|---:|
| **A4** | **叶子逆搬到 cube**（两个 32×32 三角逆用 MMAD/分块递归求逆，AIC 有 79% 余量） | §8.2：单项 2,362 µs | **~19%** |
| **A2/A1'** | `V0+V1` 剩下的 4,892 µs：gate 的 exp/softplus 链 + `V1` 三族指数（`E(G−Gref)`/`E(Gref−G)` 与 V0 重复）+ chunk 内逐行 cumsum（代码里明确为了 gk 逐位一致而保留的 63 步串行） | §8.1：仍是最大项 | ~10~20%（待再切一层确认） |
| **A1'** | GVA 下 norm 按 HK 去重（case 65 上限 ~31%，但只影响 GVA 形状） | §2.3 | 形状相关 |
| — | 调度层（V6 提前 / 省回合 / 尾部展开） | 仍非瓶颈 | 低 |

**下一轮建议**：先做 A4（叶逆外迁），同时用 `e2_sub.sh` 把 `V0+V1` 的 4,892 µs 再切一刀
（分别屏蔽 gate 的 exp 链、`V1` 的三族指数、cumsum 串行段），再决定是"合并 V0/V1 的指数"
还是"把 cumsum 换成块状 scan"。

### 8.4 再切一刀的结果：账目已经摊平，AIV 仍是主单元但不再饱和

用 `e2_sub2.sh`（块级屏蔽）+ 属性消融在**新基线 12,184.7 µs** 上把 `V0+V1` 拆开：

| 项 | 测量方式 | 时长 (µs) | 占整算子 |
|---|---|---:|---:|
| `V0+V1` 合计 | 调用点屏蔽 | 4,892 | 40.1% |
| ↳ **Q/K L2 norm（新实现）** | `use_qk_l2norm_in_kernel=False` | **1,976** | **16.2%**（改前 8,620 / 45.5%） |
| ↳ `V1`（Gref + 三族指数） | 屏蔽 V1 的 tile 循环 | 1,460 | 12.0% |
| ↳ chunk 内 cumsum（63 步串行） | 屏蔽该 for 循环 | 333 | 2.7% |
| ↳ gate 数学（exp/softplus，V0+V6 两处） | 块级屏蔽 | 299 | 2.5% |
| ↳ 余量（beta、搬运、重叠） | — | ~824 | ~7% |

**结论**：`cumsum`（333）与 gate（299）都比预期小 —— **不需要**再为它们做块状化改造；
新的块状 norm 已经把 norm 从 8.6 ms 砍到 2.0 ms（−77%），但它**仍是单项最大的向量工作之一**。

### 8.5 优化后的整核画像（msprof op，case 63）

| pipe | 改前（任务 18,812 µs） | **改后（任务 12,148 µs）** |
|---|---:|---:|
| AIV vec | 12,607 (67.0%) | **7,872 (64.8%)** |
| AIV scalar | 4,726 (25.1%) | 4,294 (35.3%) |
| AIV mte2 / mte3 | 607 / 544 | 593 / 508 |
| AIC busy | ≈3,441 (21%) | ≈3,466 (28.5%)（cube 仍只 264、fixpipe 795、mte2 748、scalar 1,287） |
| AIC `scalar_wait` | 7,594 | 4,740 |
| AIV `scalar_stall_by_ub` | 5,845 | 5,834 |
| AIV `scalar_vector_stall` | 6,208 | **864** |

读数：
1. **AIV vec 仍是主单元**（7.87 ms busy / 10.38 ms AIV elapsed），但已**不再饱和**（65%），
   所以"继续把向量活搬走/压缩"仍然是主线；
2. `aiv_scalar_vector_stall` 从 6.2 ms 掉到 0.86 ms ⇒ 逐行标量往返这一类的代价基本被消掉了，
   而 `scalar_stall_by_ub` 几乎没变（5.83 ms）⇒ 剩下的标量-UB 交互主要来自 **V3 的逐行解包
   / 叶子逆**那一类代码；
3. AIC 侧 busy 只有 3.47 ms、cube 264 µs ⇒ 任何"把活搬到 cube"的方案都有充足余量。

### 8.6 当前账目汇总（新基线 12,184.7 µs，按单项上限排序）

| # | 项 | 上限 | 备注 |
|---|---|---:|---|
| 1 | **V3 的两次 32×32 叶子逆**（AIV、串行） | **2,362 (19.4%)** | 已用 `e2_sub.sh` 单独屏蔽验证；对应 A4 |
| 2 | `C4+C5+C7`（cube） | 2,246 (18.4%) | AIC 有大量余量，属"等待/串行"而非吞吐 |
| 3 | `V6`（AIV 块指令） | 2,002 (16.4%) | |
| 4 | `C2`（cube） | 1,988 (16.3%) | |
| 5 | **新版 norm（AIV）** | 1,976 (16.2%) | 还能靠"双 buffer 免二次 Cast"再砍 ~30% |
| 6 | `V1`（AIV 三族指数） | 1,460 (12.0%) | 与 V0 的 gate 指数有重复计算 |

**下一项的具体设计已单独成文**：`kda_prepare_a2a3_A4_leaf_inverse_to_cube.md`
（A4：把两次 32×32 叶子逆从 V3 外迁到新的 cube stage `C3`，含公式、分块幂级数算法、
space/同步预算、回合代价估算与验证计划；并给出"不新增 stage 的备选：把 62 次 Gather
降为 2 次 + 屏障减半"）。

### 8.7 叶子逆"合并两叶、共享屏障"落地实测（A4 的 §6 备选第一步）

实现：`patch_leafpair.py` + `leafpair_helper.txt`（在 A1 候选 `vec_normblk4.h` 之上）。
两个叶子互不依赖，把原来两次 `SolveLeafRightLooking`（2 × 31 步 × 3 屏障 = **186 个屏障**）
合并成**一次 31 步循环、93 个屏障**，两条独立链在 vector pipe 里互相掩盖；
每步算子与顺序不变（Gather→Mul→Sub），两条链各自读写自己的 X。

| 版本 | case 63 时长 | 相对基线 |
|---|---:|---:|
| 原始基线 | 18,851 ~ 18,832 µs | — |
| A1（块状 norm） | 12,188.6 µs | −35.3% |
| **A1 + 叶逆合并（`leafpair`）** | **11,922.6 µs** | **−36.7%** |

精度：`acc_batch.sh` 跑 13/26/60/61/62/68/80 **7/7 通过，且误差与上一版逐位相同**
（1.192e-07 / 2.384e-07 / 4.883e-04 / 9.766e-04 / 4.883e-04 / 9.766e-04 / 4.883e-04）
⇒ 该改动是**数值等价**的。

**由这次实测得到的定量结论（很重要）**：

```text
叶子逆单项            = 2,362 µs
屏障减半（186→93）只省 =   266 µs   ⇒ 屏障只占叶子逆的 ~22%
剩余 ~1.8 ms 落在       = 62 次/head-chunk 的 Gather + 每步的短向量 Mul/Sub
```

所以 **A4 的 §6 备选（"把 Gather 降为 2 次"）不再是首选**：
真正值钱的是"换执行单元"——cube 版本（A4′）根本不需要 Gather，
2,362 µs 的绝大部分会直接消失；这也把 §8.6 表里第 1 行的优先级进一步坐实。

**再拆一层（`patch_nogather.py`：只关掉每步 Gather、保留 Mul/Sub）**：

```text
11,905.0 µs (lshadow，含 Gather)  →  10,640.7 µs (nogather)   差 −1,264 µs (−10.6%)

⇒ 叶逆 2,362 µs = Gather(62 次/head-chunk，广播系数列) 1.26 ms
                  + 短向量 Mul/Sub + 屏障              1.10 ms（其中屏障只 ~0.27 ms）
```

两块在分块矩阵乘（A4′）里都天然消失 ⇒ A4′ 的收益上限有明确来源，
"在 AIV 上继续抠叶逆"的上限很有限（Gather 换布局会牵动 prod 重复参数与 x 行主序）。
