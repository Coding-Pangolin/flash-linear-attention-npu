# pre_process_fwd_kernel_merged 设计输入（v2）

> **文档性质**：03 方案设计的输入参考，不是流程产物。
> 流程产物是算子工程里的 `docs/api.md`（01）、`reference/reference.py`（02）、`docs/design.md`（03）。
>
> **与 v1 的关系**：本文取代 `pre_process_fwd_kernel_merged_design_analysis.md`（2026-09-19）。
> v1 的整体结构（仿射变换推导、m 为什么需要高精度、shape 流转、R20 分析）经核实**正确，继续沿用**；
> 但 v1 有 **3 处被后续实测推翻或修正**，在 §7 逐条列出。
>
> 整理日期：2026-09-20

---

## 0. 结论摘要

| # | 结论 | 依据 |
| --- | --- | --- |
| 1 | H20 性能基线（单次 kernel 调用，T=11264/HK=HV=32/K=V=128）**取决于 precision 档位**：`default`(tf32) **1620.7 µs**、`tf32x3` **3404.7 µs**、`ieee` **18321.6 µs** | §3.1，实测 |
| 2 | **m 半边是瓶颈**（default 占 65%，ieee 占 90%），且它跨档位差 **41×**，全部来自 `M_c @ m` 一个 FP32 乘 | §3.2 |
| 3 | **两半在 H20 上没有并行**（`both ≈ h + m`，不是 `max(h,m)`）→ "两条独立链可并行"的前提不成立 | §3.3 |
| 4 | 950PR 上 FP32 matmul 不开 HF32 是 **`cube_k=1`**（BF16 的 1/16），开 HF32 是 `cube_k=8`（BF16 的 1/2）。实测：大 GEMM 上 HF32 加速 **6.02×**、达成率 **88% / 66%**（原生 / HF32） | §2.2 文档 + §2.4 实测 |
| 5 | ⭐ **第一取舍是组件粒度，不是精度路径。** 单次 CATLASS `DeviceGemm` 调用在 `128×64×128` 上要 **6.0–8.1 µs**，其中 **99% 是固定开销**（纯 Cube 只要 0.011–0.089 µs）；按"每 chunk 一次调用"外推，单核约 **4.8 ms**，而吞吐下界只有 **281 µs** —— **差 17 倍**。所以 chunk 循环必须留在核内、Cube 用 **Tile 级**组件手写 | §2.4 / §5.2 / §5.5 |
| 6 | HF32 与原生 FP32 在我们形状上只差 **1.35×**（被固定开销掩盖），**不是决定性因素**；精度路径的取舍要等核内循环结构定下来之后再谈 | §2.4 / §5.2 |
| 7 | 按 roofline，**上游式在 H20 是计算受限、在 950PR 是带宽受限**（算力/带宽比差 6.8×）→ **V 维建议合并** | §5.1 |
| 8 | H20 侧标杆对齐已通过（h_half 1.000000 / m_half 严格复核 1.000000） | §3.5 |

---

## 1. 算法结构（沿用 v1，已核实）

`chunk_gated_delta_rule_fwd_h` 的逐步状态更新是一个**仿射变换**：

```text
h_{c+1} = decay_c·h_c + k_cᵀ(v_c − w_c·h_c) = (decay_c·I − k_cᵀw_c)·h_c + k_cᵀ v_c
        =              M_c              · h_c +  L_c
```

展开 `NT` 步即 `S_out = m·S_in + h`，其中

- `m = Π M_c`（齐次部分，**乘积链**）
- `h = Σ (partial products)·L_c`（特解，**衰减累加**，等于初值取 0 的结果）

这与上游 kernel stage 1 的行为一致（`b_h1` 初值 0，逐 chunk `*= decay`、`+= kᵀv_new`）。

本算子输出 `hm[HV, K, V+K] = [h | m]`，上游 CP 包装据此做跨卡前缀复合得到每个 rank 的
`initial_state`，再交给原有的 `chunk_gated_delta_rule_fwd_h`。跨卡 `all_gather` 与
`merge_fwd_bwd_kernel` **不在本算子范围内**。

粒度：**一个窗口（一个 part）**。上游 CP 包装对 front/back 各调一次 kernel。

---

## 2. 昇腾侧硬件事实

### 2.1 平台参数（Ascend950PR_9579，读自 `docs/precheck.md`）

| 项 | 值 |
| --- | --- |
| `NpuArch` | 3510（编译 `--npu-arch=dav-3510`） |
| AIC / AIV | **28 / 56**（Cube : Vector = 1 : 2） |
| 频率 | 1.65 GHz |
| L0A / L0B / L0C | 64 KiB / 64 KiB / **256 KiB** |
| L1 / UB | 512 KiB / 248 KiB（每 AIV） |
| L2 / HBM | 128 MiB / 约 1.5 TB/s |
| BF16 峰值算力 | 约 378 TFLOPS（28 核档） |

### 2.2 FP32 matmul 的硬件实现（决定性依据）

来源：`asc-devkit/docs/zh/api/SIMD-API/c_api/cube_compute/asc_enable_hf32.md`（官方文档）

**表 2** HF32 对 Mmad 理论性能的影响（NPU 架构版本 3510）：

| 接口 | 左矩阵 A | 右矩阵 B | `cube_m` | `cube_n` | **`cube_k`** | `k_0` |
| --- | --- | --- | --- | --- | --- | --- |
| `asc_mmad`（**不开启** HF32） | float | float | 16 | 16 | **1** | 8 |
| `asc_mmad`（**开启** HF32） | float | float | 16 | 16 | **8** | 8 |

即每 AIC 每拍：

| 模式 | MAC/拍 | 相对 BF16 |
| --- | --- | --- |
| BF16（`cube_k=16`） | 4096 | 1× |
| **FP32 + HF32** | **2048** | **1/2×** |
| **FP32 原生（不开 HF32）** | **256** | **1/16×** |

文档原文：*"针对 Ascend 950PR/Ascend 950DT 产品，开启 HF32 可使 Mmad（f322f32）接口的
计算性能提升至原来的八倍。"* 且 **950PR 的 HF32 尾数为 10 位**（与 NVIDIA TF32 同级）。

**对本算子的直接含义**：`M_c @ m`（M=K，N=BS，Kdim=K）每步的 Cube 拍数：

| 实现路径 | 每步拍数 | 精度 | 说明 |
| --- | --- | --- | --- |
| FP32 原生（不开 HF32） | **4096** | 真 FP32 | 最准、最慢 |
| HF32 单遍 | **512** | ~TF32（10 位尾数） | 链长 176 时误差 ~1e-1，**不可接受** |
| HF32 拆分 3 遍 | 1536 | ~20 位 | 需 Vector 侧拆分 |
| BF16 拆分 3 遍 | **768** | ~16–21 位 | 需 Vector 侧拆分，拍数最省 |

对应 m 半边每步总拍数（含 `kᵀw` 的 256 拍）：**4352 / 768 / 1792 / 1024**，
而 h 半边每步固定 256 拍。**这个选择直接决定 m : h 是 17×、3×、7× 还是 4×。**

### 2.3 CANN 版本差异（做实验时必须知道）

`asc-devkit` 示例与文档对应 **CANN ≥ 9.2.0**；当前容器是 **CANN 9.1.0**，ASC C API 不同：

| 文档/示例（≥9.2.0） | 本机 9.1.0 实际 |
| --- | --- |
| `asc_enable_hf32()` / `asc_disable_hf32()` | `asc_enable_hf32()` / `asc_set_fp32_mode()` |
| `asc_set_hf32_round_mode(asc_hf32_round_mode::…)` | 无 |
| `asc_mmad(..., asc_unit_flag_mode::DISABLE, bool, bool, bool)` | `asc_mmad(..., uint8_t unit_flag, bool, bool, bool)` |
| `asc_mmad(c, a, b, m, k, n, …)` | `asc_mmad(c, a, b, left_height, n_dim, right_width, …)` |

裸 `asc_mmad` 那条路走不通：三套 ASC C-API 互不兼容，而 CANN 9.2.0-beta.2 又不自带匹配的示例，
手推 5 个搬运接口的参数风险太高。**改走 CATLASS（见 §2.4）**，实测已完成。

### 2.4 CATLASS fp32 matmul 实测（2026-09-20）

选 CATLASS 的理由有三：仓库里就有可编译的工程；**本算子最终也走 CATLASS**，量出来的数更贴近设计；
而且 `examples/68_ascend950_multi_core_splitk_matmul` 就是 float×float→float，
且 **HF32 是显式模板参数**：

```cpp
using ElementA = float; using ElementB = float; using ElementC = float;
constexpr bool useHF32 = false;                                        // ← 可直接切换
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
```

改写为 `examples/99_bench_fp32_mmad`（加 200 次循环计时，两档 HF32 在同一次运行里对比）。
环境：Ascend950PR_9579 / CANN 9.1.0 / 28 AIC / 1.65 GHz；`ASCEND_RT_VISIBLE_DEVICES=0`。

| 形状 | FP32 原生(HF32=off) | FP32(HF32=on) | HF32 加速 |
| --- | --- | --- | --- |
| **128×64×128（本算子 `M_c@m`）** | **8.092 µs** | **5.997 µs** | **1.35×** |
| 128×128×128 | 10.899 µs | 6.758 µs | 1.61× |
| 128×64×8192 | 15.456 µs | 10.263 µs | 1.51× |
| 512×512×512 | 30.315 µs | 13.764 µs | 2.20× |
| **2048×2048×2048** | **827.5 µs**（10.4 TMAC/s） | **137.4 µs**（62.5 TMAC/s） | **6.02×** |

对照 §2.2 的理论值（原生 11.8 TMAC/s、HF32 94.6 TMAC/s）：**达成率 88% / 66%**。

**三条结论：**

1. **文档说的 8× 是真的** —— 大 GEMM 上实测 6.02×，不是文档写错。
2. **但在本算子的形状上完全看不见这 8×。** `128×64×128` 的纯 Cube 只需 **0.089 µs**（原生）/
   **0.011 µs**（HF32），实测却是 **8.09 / 6.00 µs** —— **99% 是固定开销**
   （kernel launch + tiling + L1/L0 staging + 同步）。
3. 所以 **HF32 与原生只差 1.35×**；而"每 chunk 一次 matmul 调用"会让总时间比吞吐下界高 **17 倍**
   （见 §5.5）。

原始数据：`scripts/bench_fp32_mmad_catlass_result.txt`；程序：`scripts/bench_fp32_mmad_catlass.cpp`。

---

## 3. H20 实测基线

采集脚本 `benchmarks/cp/bench_pre_process_h20.py`，case `model-gk`：
`T=11264, HK=HV=32, K=V=128, BT=64, grid=(4,32)`，event p50。

### 3.1 整体

| precision | 单次 kernel | 有效算力 |
| --- | --- | --- |
| `default`（NVIDIA 上是 **TF32**） | **1620.7 µs** | 43.7 TFLOPS |
| `tf32x3` | 3404.7 µs | 20.8 TFLOPS |
| `ieee`（真 FP32） | 18321.6 µs | 3.9 TFLOPS |

H20 峰值 BF16 约 148 TFLOPS（78 SM），即最高只跑到 **30%**。

**`default` 与 `tf32x3` 差 2.1×，与 `ieee` 差 11.3×。所以"模型实际用哪个档位"是基线的前置条件。**

### 3.2 半边拆解

| precision | h (µs) | m (µs) | h+m | both | both/max | both/sum |
| --- | --- | --- | --- | --- | --- | --- |
| default | 388.1 | 1058.1 | 1446.2 | 1620.7 | 1.53 | 1.12 |
| tf32x3 | 392.8 | 2564.4 | 2957.2 | 3404.7 | 1.33 | 1.15 |
| ieee | 401.7 | 16465.2 | 16866.9 | 18321.6 | 1.11 | 1.09 |

三条读数：

1. **m 是瓶颈**：default 占 65%，ieee 占 90%。m 的 MAC 只有 h 的 2.00×，耗时却是 2.73×（default）。
2. **h 对 precision 几乎不敏感（+3.5%）**：它两个 dot 的输入都是 BF16（`h`、`v_new` 都降到 bf16），
   `AFFINE_CHAIN_PRECISION` 只作用在 `M_c@m`。→ **h 半边在昇腾上没有 FP32 计算压力。**
3. **m 跨档位差 41×**，全部来自那一个 FP32 乘。

### 3.3 两半没有并行（**v1 的关键错误**）

`both ≈ h + m`，而不是 `max(h,m)`：比值 1.11–1.53 倍 max、1.09–1.15 倍 sum。
128 个 program 铺在 78 个 SM 上只有 1.64 波，两半实际上在争同一批核。

**昇腾只有 28 个 AIC，128 个 program 要 4.57 波，更不可能并行。**
所以"两条独立链能并行铺在不同核上"这个前提**在两边都不成立**。

### 3.4 带宽利用率（决定 V 合并的关键数据）

由 `both` 反推：逻辑读约 900 MB / 1620.7 µs = **约 555 GB/s**，而 H20 的 HBM 是 **4.0 TB/s**
→ **带宽利用率只有 14%**。

**所以 H20 上（含 3× 的 k/w 重复读）根本不是带宽受限，"合并省带宽"在 H20 上完全看不出价值。**

### 3.5 精度对齐结果（02 阶段收口）

`--precision ieee` 采一次，与 CPU 标杆对齐：

```
region        elements   matched     err     max_abs
ALL            1048576  1.000000       0   9.526e-03
h_half          524288  1.000000       0   9.526e-03
m_half          524288  1.000000       0   3.465e-07
[严格复核] m_half 用 atol=1e-6 复算: matched=1.000000 max_abs=3.465e-07
```

`m_half` 在 `atol=1e-6` 下仍全过，说明 m 的公式、正负号、FP32 链式累加完全正确。
`h_half` 的 9.5e-3 是 `bf16(h)` 量化不连续在反馈环里的**内在散布**（实测：同为 FP32、
仅把 token 累加按 16 分组，h 半边就散布 7.2e-3；m 半边恒为 0）。

由此校准的 float32 策略：`atol=1.5e-2 / rtol=2e-3 / max_abs_limit=0.05`。
**h 半边是弱检查、m 半边是强检查**，这条结论要带进 05 的测试计划。

---

## 4. Stage 划分的规则依据

### 4.1 cannbot Skill（R01–R21）

唯一直接提到 chunk 依赖的是 **R20**：*"chunk 间无依赖时 `Nbase` 为本次调用的 chunk 总数；
有依赖时 `Nbase` 为 sequence 总数"* → **chunk 依赖直接砍掉一个并行维度**。
本算子 `B=1`，故 `Nbase=1`、`Nwork=ceil(HV/CG)`，`blockDim` 上限就是 `ceil(HV/CG)`。

其余相关条款：R01（先按数据依赖/计算类型/生命周期/精度观察点划 Stage，再分配 AIC/AIV）、
R02、R13（无依赖的 Cube/Vector Stage 按可并行建模，存活数据用互不重叠区间）、
R15（优先合并连续同类操作）、R19（先算同时存活容量；容量不足先评估 GM 中转，
只有仍解决不了正确性/生命周期/性能才加 Stage）。

**Skill 的空白**：`solution-design-reference.md` 唯一的完整示例
（`catlass_chunk_gated_product`）明确是"**chunk 之间独立**"，没有 chunk 依赖的范例。

### 4.2 仓库自己的流程（`docs/agents/`）

仓库把 chunk 依赖当一等分支，给出参考算子 `ChunkGatedDeltaRuleFwdH` 的 4 个 Stage：

| Stage | 单元 | 计算 |
| --- | --- | --- |
| S0 | Cube | `P = w @ h_prev` |
| S1 | Vector | `v_new = u − P` + gate |
| S2 | Cube | `delta_h = k_or_kgᵀ @ v_new` |
| S3 | Vector | `h_next = gate_last·h_prev + delta_h` |

调度约束：同序列按依赖顺序；不同序列/head 可并行；每核按 head round，每轮最多 4 head；
当前 chunk 全部 head round 完成状态更新后才进下一 chunk；两套窗口 ping-pong 共 8 slot；
四类跨核状态 `cube1Done/vec1Done/cube2Done/vec2Done`。

**03 的合理做法**：以 S0–S3 为起点，逐条过 R01–R21 证明它在当前场景
（`B=1`、多出 m 半边、h 与 m 不并行）下成立，推导过程写入 `docs/design.md` 第 6 章。

### 4.3 h / m 两侧的依赖性质

| | h 半边 | m 半边 |
| --- | --- | --- |
| 依赖类型 | **反馈递推**（`h → v_new → kᵀv_new → h`） | **乘积链**（只有 `M_c @ m` 串行） |
| 关键路径 | 该 chunk 几乎全部计算 | 只有 `M_c @ m` |
| 可预算部分 | 几乎没有 | `kᵀw`、`diag`、`M_c` **全部与 m 无关，可提前算** |
| 并行度来源 | head × V 列块 | head × K 列块 + 预算部分 |

**m 半边的 `M_c` 与链无关** —— 这是昇腾上最有价值的一条优化线索：可以先把所有 `M_c` 预算出来，
再用流水的方式做链式乘，把串行链压到只剩乘法本身。

---

## 5. 关键设计判断

### 5.1 V 维是否合并：**建议合并**（昇腾口径）

先做 roofline：

| | BF16 峰值 | HBM | 脊点 (FLOP/byte) |
| --- | --- | --- | --- |
| H20（78 SM） | 148 TFLOPS | 4.0 TB/s | **37** |
| **950PR（28 AIC）** | **378 TFLOPS** | **1.5 TB/s** | **252** |

**昇腾的算力/带宽比是 H20 的 6.8 倍。** 两个方案的算术强度：

| 方案 | 运算 | 读入 | 强度 |
| --- | --- | --- | --- |
| 上游式（k/w 读 4 遍） | 70.87 GFLOP | 830.5 MB | **85.3** |
| 合并式（k/w 读 1 遍） | 59.06 GFLOP | 276.8 MB | **213.4** |

对照脊点：

- **H20**：85.3 > 37 → 计算受限；213 > 37 → 计算受限。**合并没有带宽收益**，且实测两半本就串行，
  所以 H20 上看不出合并的价值（实测也印证：带宽利用率仅 14%）。
- **950PR**：85.3 < 252 → **带宽受限**；213 < 252 → 接近脊点、基本回到计算受限侧。

折算下界：

| | Cube 吞吐下界 | DRAM 下界 | 综合下界 |
| --- | --- | --- | --- |
| 上游式 | 312 µs | **553.6 µs** | **553.6（带宽）** |
| 合并式 | 280.9 µs | 184.5 µs | **280.9（Cube）** |

**昇腾上合并的下界约为上游式的一半。** 这个结论只来自硬件参数，与 H20 实测不冲突 ——
H20 看不出来，正是因为它算力/带宽比只有昇腾的 1/6.8。

**合并的代价**（必须写进设计并在 04 验证）：

1. 串行链从 136.5 µs 变成 245.8 µs（每步 3 个 dot: `w@h → kᵀ[v|w] → M@m`），
   离吞吐下界 280.9 µs 只剩 14% 余量，**几乎没有流水气泡空间**。
2. 片上容量：K=256 时 `M_c` 是 256 KB，超过 L0A（64 KB）、逼近 L0C（256 KB），**必须切 K 维**。
3. DPLR 有第三项 `bgᵀ @ v_new`，左因子与 `kᵀ` 不同，拼不进同一次乘。

### 5.2 取舍的优先级：**组件粒度优先于精度路径**（本节结论已被实测修正）

> ⚠️ v2 初稿把 `M_c@m` 的精度路径当作首要取舍；2026-09-20 的实测（§2.4）推翻了它。

按 §2.2 的理论拍数，精度路径看起来是决定性的（m:h 从 17× 到 3×）。但 §2.4 实测表明：

- 在**本算子的形状**上，一次 CATLASS matmul 调用 **99% 是固定开销**，
  HF32 与原生只差 **1.35×**；
- 真正的量级差异来自 **"每 chunk 一次调用"还是"chunk 循环留在核内"** —— **17×**（§5.5）。

所以优先级是：

1. **第一：组件粒度**（§5.5）—— 决定固定开销被摊薄多少，量级差 17×；
2. **第二：Stage 划分与流水**（§4）—— 决定串行链能否被盖住；
3. **第三：`M_c@m` 的精度路径** —— 只有在正确的粒度下，§2.2 那几个拍数才是有效比较。

精度路径本身的目标不变：rtol 2e-3 + 链长 176 → 相对误差 ≲ 1e-3 → 约 **20 位有效尾数** →
候选是 **BF16 3 遍拆分**或 **HF32 3 遍**；**HF32 单遍**（10 位尾数，链长 176 → ~1e-1）不可行。
但它要**等粒度定下来、用核内循环实测单步成本之后**再选。

### 5.3 可选的第三条路：前缀扫描

由 §1 的展开，整个窗口是 NT 张仿射映射的有序复合，而复合可结合：
`(M_B,L_B)∘(M_A,L_A) = (M_B M_A, M_B L_A + L_B)`。因此串行链可压到 `~2·log₂NT` 层（176 → 约 15 层）。

代价（每 chunk 每 head，估算）：算力 +20%（5.25M → 6.3M MAC），需缓存 `NT × K×K` fp32
（单 head 11 MB，32 head 约 **360 MB HBM**）。

**在昇腾带宽受限的前提下，360 MB 的额外 HBM 流量是硬伤**（1.5 TB/s 下约 240 µs），
所以这条路只在 `HV` 很小、串行链真正成为瓶颈时才值得评估。

### 5.4 消除 `M_c` 的重复计算

上游每个 m program 都算完整的 K×K `M_c`（K=128、BS=64 时算 2 遍），但 `M_c` 与列块无关。
昇腾上"重算一个小矩阵"通常比"GM 往返共享"便宜，所以要先量化：
省下的 Cube 拍数 vs GM 中转的字节数。**这条不受 V 合并决策影响，是独立的优化项。**

### 5.5 组件粒度：**用 Tile 级手写，不用 `BlockMmad` / `Kernel` / `DeviceGemm`**

#### 5.5.1 定量依据

§2.4 实测：单次 `DeviceGemm` 调用在 `128×64×128` 上 **6.0–8.1 µs**。若每个 chunk 调一次：

```text
单核 program 数      = ceil(128 个 program / 28 AIC) ≈ 4.57
每 program chunk 数  = NT = 176
单核调用次数         ≈ 804
总时间               ≈ 804 × 6 µs ≈ 4.8 ms
```

而 §5.1 算出的 Cube 吞吐下界是 **281 µs** —— **差 17 倍**。所以 **chunk 循环必须留在核内**，
让 launch / tiling / staging / 同步这些固定开销只付一次。

#### 5.5.2 workflow 里的依据（有两处说法不一致，按"完整示例 + 仓内先例"执行）

| 出处 | 说法 |
| --- | --- |
| 插件 `AGENTS.md` **G4** | *"op_kernel 直接 `Kernel{}(params)`，禁用 `DeviceGemm`；**禁用自实现矩阵乘/逐元素/拷贝循环**"* —— **但 G4 属于 legacy 路由（Step 1–7）**；LA 专用流程的对应约束是 **G6**，交给 LA skill 五阶段，**不继承 G4** |
| LA skill `stage-design-rules.md` **第 7 行（术语）** | *"Cube 操作……使用 CATLASS `BlockMmad`/Kernel 组件"* |
| LA skill `stage-design-rules.md` **第 182 行（设计清单 6）** | *"CATLASS ArchTag、DispatchPolicy、**TileShape、BlockMmad、BlockEpilogue、BlockScheduler、Kernel**……"* |
| LA skill `solution-design-reference.md` **§2.5/§2.6（完整设计示例）** | 伪代码是 `LocalLoadQK()` / **`LocalMmadScore()`** / **`LocalMmadOutput(r)`**；并明确 *"`LocalMmad*` **管理同一套物理 L0 缓冲**。每个 L0A/B 槽由 MTE1 写、Cube 读；每个 L0C 槽由 Cube 写、Fixpipe 读。前者按 `MTE1_M/M_MTE1`、后者按 `M_FIX/FIX_M` 闭环复用"*；tile 形状是 `[64,64,128]` / `[64,128,64]` |

**skill 自己的完整示例就是"手写 tile + 自管 L0 槽 + 自配 HardEvent"**，比 `BlockMmad` 低一层。
术语定义与设计清单第 6 项把层级写在 `BlockMmad` 上，与示例不一致；处理方式是在
`docs/design.md` 里显式声明采用的组件层级并给出依据（见 5.5.4）。

#### 5.5.3 仓内先例（最强证据）

`chunk_fwd_h` —— **Ascend950 上已验证、537 µs / 目标 573 µs 达标**的那个算子 ——
`op_kernel/arch35/chunk_fwd_h_cube.h`（797 行）：

```cpp
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"      // ← Tile 级
#include "catlass/gemm/tile/tile_mmad.hpp"      // ← Tile 级
using TileMmadS0 = Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t, ...>;
...
AscendC::LocalTensor<...> L0A(uint32_t slot);   // 自己管 L0 槽
AscendC::LocalTensor<...> L0C(uint32_t slot);
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));   // 自己配事件
```

**没有 `BlockMmad`、没有 `BlockScheduler`、没有 `DeviceGemm`、没有 `Kernel{}`** —— 只借用
catlass 的 `arch` / `layout` / `tile_copy` / `tile_mmad` / `tla`，外加仓库自己的
`common/kernel_utils/tile/copy_l0c_to_ub.hpp`。

#### 5.5.4 写进 `docs/design.md` 的说法

> 本算子 Cube 部分采用 CATLASS **Tile 级**组件（`Gemm::Tile::TileMmadTla` + `TileCopy`），
> 不使用 `BlockMmad` / `BlockScheduler` / `Kernel` / `DeviceGemm`。依据：
>
> 1. 本算子的矩阵乘是 **chunk 内小尺寸、chunk 间串行的递推**（`M=K≤256, N=BS=64, Kdim=BT=64`），
>    单个 GEMM 远小于常规 `L1TileShape`（如 256×256×128）；
> 2. 实测（§2.4）单次 `DeviceGemm` 在 `128×64×128` 上耗时 6.0–8.1 µs，其中 99% 为固定开销；
>    按 chunk 调用外推单核约 4.8 ms，而 Cube 吞吐下界 281 µs，**差 17 倍**；
> 3. chunk 循环必须留在核内，L1/L0 槽与 HardEvent 由本算子按 Stage 显式管理 ——
>    与 `references/solution-design-reference.md` §2.5/§2.6 的 `LocalMmad*` 示例一致；
> 4. 仓内已验证算子 `chunk_fwd_h` 采用同一层级（`TileMmadTla` + 自管 L0 槽 + `AscendC::HardEvent`），
>    作为本算子的实现范式。

---

## 6. 未决问题清单（进入 03 前需要的信息）

### A. 必须由用户提供

| # | 信息 | 影响 |
| --- | --- | --- |
| 1 | 模型实际是否开 `use_tf32x3_affine_chain` | **决定 H20 基线是 1620.7 µs 还是 3404.7 µs（2.1×）** |
| 2 | 对标 case 的 CP 配置（world_size）→ 每 rank 窗口长度 `T_win` | 现在测的是 `T=11264`（等价非 CP）；耗时近似线性于 `NT`，这个不定基线就定不了 |
| 3 | 1.0× 的口径：单次 kernel 调用，还是"一个 rank 的整个 pre_process"（front/back 两次 + gather + merge） | 决定目标值是 1× 还是约 2× |
| 4 | 每个模型 case 的完整 shape/dtype（`HK/HV/T/K/V/BT`、是否 varlen） | 03 的 tiling 与工作量推导 |
| 5 | 性能统计口径（预热/采样次数，是否 `msprof` `op_summary` 的 `Task Duration`） | 05 的验收判定 |

### B. 需要在昇腾上实测/确认

| # | 事项 | 状态 |
| --- | --- | --- |
| 6 | 单次 CATLASS matmul 调用在本算子形状上的代价 | **已完成**（§2.4）：`128×64×128` 上 6.0–8.1 µs，99% 是固定开销 |
| 6a | ⭐ **核内 chunk 循环的单步成本** —— 把 176 个 chunk 放进一个 kernel、共享 tiling/staging/同步之后的真实单步耗时 | **新增，未做。这是当前最关键的一个缺口。** 需要写一个 `TileMmad` 级的核内循环基准（不是 `DeviceGemm` 逐次调用），它直接决定 §5.5 那 17× 的收益能否兑现 |
| 6b | `M_c@m` 的精度路径在核内循环下的真实比例（FP32 原生 / HF32 / 拆分） | 待 6a。现在的 1.35× 是被固定开销掩盖的，只有在核内循环下才看得出真实比例 |
| 7 | `kᵀw` 用 BF16 是否满足精度（它是单步乘，理论上够） | 03 用仿真确认 |
| 8 | K=256 时 `M_c` 的切分方案（容量超限是硬约束） | 03 |
| 9 | 消除 `M_c` 重复计算的收益（重算 vs GM 共享） | 03 |

### C. 已可定稿

| # | 项 | 结论 |
| --- | --- | --- |
| 10 | V 维是否合并 | **建议合并**（§5.1，按昇腾带宽论证），03 需按 R13/R19 落成容量与 Stage 证据 |
| 11 | Stage 划分起点 | 以仓库 chunk-dependent 的 S0–S3 为起点，逐条过 R01–R21（§4.2） |
| 12 | `h` 半边的计算精度 | 两个 dot 都是 **BF16 输入**，无 FP32 压力（§3.2 实测印证契约里的两个 bf16 舍入点） |
| 13 | 精度策略 | `atol=1.5e-2 / rtol=2e-3 / max_abs_limit=0.05`（§3.5） |
| 14 | ⭐ **组件粒度** | **用 CATLASS Tile 级（`Gemm::Tile::TileMmadTla` + `TileCopy`）手写，chunk 循环留在核内，L1/L0 槽与 HardEvent 自己管；不用 `BlockMmad` / `BlockScheduler` / `Kernel` / `DeviceGemm`**（§5.5，定量依据 + workflow 依据 + 仓内先例三条齐备） |

---

## 7. 对 v1 分析文档的修正

| v1 的说法 | 实际 | 依据 |
| --- | --- | --- |
| §3.1 / §5.5："h 与 m 数据依赖独立，**可以完全并行**"；"128 条独立链 vs 28 个核，依赖不是吞吐瓶颈、能被其他链盖住" | **两半并没有并行**：实测 `both ≈ h + m`（1.12 倍 sum、1.53 倍 max）。设计必须按"两半争同一批核、串行推进"来算 | §3.3 实测 |
| §2.5 / §7.4："V 维合并不确定，需要实测和基线" | 现在可以定：**建议合并**。理由不是照抄 H20，恰恰相反 —— 昇腾的算力/带宽比是 H20 的 6.8 倍，上游式在昇腾上**带宽受限**（强度 85.3 < 脊点 252） | §5.1 |
| §5.4 前缀扫描"深度从 176 降到 ~15 层，算力多约 20%" | 结论数字对，但**漏算了 360 MB 额外 HBM 流量**（约 240 µs @1.5 TB/s），在带宽受限的前提下这条路基本走不通 | §5.3 |
| §2.3 "TF32 尾数 10 位 → 链长 176 时 ~0.09" 的估算 | 估算方法正确；**现在有了昇腾侧的对应事实**：950PR 的 HF32 也是 10 位尾数，且 cube_k 从 16 降到 8 | §2.2 |
| §8 未决问题 #1/#2（基线、HV） | **已解决**：基线见 §3.1，`HV=32` | §3 |
| §8 #8（m 精度曲线） | 部分完成：标杆侧已量化契约舍入的影响（`docs/validation.md`）；**硬件侧待实验** | §2.2 / §5.2 |

**对本文 v2 初稿自身的修正（2026-09-20 实测后）：**

| v2 初稿的说法 | 实际 | 依据 |
| --- | --- | --- |
| §5.2 把 "`M_c@m` 的精度路径（HF32 / 拆分 / 原生）" 当作**首要取舍**，并给出 m:h 从 3× 到 17× 的拍数比较 | **首要取舍是组件粒度，不是精度路径。** 在本算子形状上单次 matmul 调用 99% 是固定开销，HF32 与原生只差 1.35×；而"每 chunk 一次调用"比"核内循环"差 17×。精度路径的拍数比较只有在核内循环下才有效 | §2.4 实测 / §5.2 / §5.5 |
| §2.3 结尾："理论数据已经足够支撑设计决策，实验的作用是实测验证" | 实测反而**改变了结论的优先级**（上一条）。理论拍数没错，但它不是主导项 | §2.4 |
| 对插件 `AGENTS.md` G4 *"禁用自实现矩阵乘/逐元素/拷贝循环"* 未加区分 | **G4 只约束 legacy 路由**；LA 专用流程的约束是 G6。LA skill 自己的完整设计示例（`LocalMmad*` + 自管 L0 槽 + 自配 HardEvent）和仓内达标算子 `chunk_fwd_h`（`TileMmadTla`，无 `BlockMmad`/`Kernel`/`DeviceGemm`）都采用 Tile 级手写 | §5.5.2 / §5.5.3 |

**v1 中继续有效的部分**：§1（仿射变换推导）、§2.1–2.4（m 为什么需要高精度）、
§3.2–3.6（`kᵀ` 的代数必然性、两次乘不同、输入搬运冗余、可合成一次 `N=V+K` 乘）、
§4（shape 流转）、§5.1–5.3（依赖性质）、§6（R01–R21 与仓库流程的条款梳理）。

---

## 附录：文件与来源

| 类别 | 位置 |
| --- | --- |
| 上游参考实现 | `fla-org/flash-linear-attention@e52dbc0e`，`fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` |
| 接口契约（01） | 算子工程 `docs/api.md` |
| CPU 标杆（02） | 算子工程 `reference/reference.py`、`reference/definition.json` |
| 开发期验证记录 | 算子工程 `docs/validation.md` |
| H20 采集与对齐 | 分支 `bench/cp-pre-process-h20`：`benchmarks/cp/bench_pre_process_h20.py`、`benchmarks/cp/pre_process_h20/` |
| 昇腾设计模型 | 算子工程 `scripts/estimate_ascend_design.py`（§[2] 串行/吞吐下界、§[3] 片上容量、§[4] 流量、§[7] 半边对照） |
| **fp32 matmul 实测（采用）** | 算子工程 `scripts/bench_fp32_mmad_catlass.cpp` + `scripts/bench_fp32_mmad_catlass_result.txt`；上游工程为 `catlass/examples/99_bench_fp32_mmad/`（改写自 `68_ascend950_multi_core_splitk_matmul`）。复跑：`bash scripts/build.sh 99_bench_fp32_mmad -DCATLASS_ARCH=3510`，再 `ASCEND_RT_VISIBLE_DEVICES=0 ./output/bin/99_bench_fp32_mmad 128 64 128 0` |
| fp32 matmul 实测（**放弃**） | 算子工程 `scripts/bench_fp32_mmad.asc` —— 裸 `asc_mmad` 路线。三套 ASC C-API（容器 9.1.0 / 宿主 9.2.0-beta.2 / asc-devkit master）互不兼容，CANN 也不自带匹配示例，保留作记录 |
| 仓内实现范式 | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h`（`TileMmadTla` + 自管 L0 槽 + `AscendC::HardEvent`，无 CATLASS Block/Kernel 组件） |
| 组件层级的规则依据 | `ops/catlass-linear-attention-workflow/references/solution-design-reference.md` §2.5/§2.6；`stage-design-rules.md` 第 7/182 行；插件 `AGENTS.md` G4（仅 legacy） |
| HF32 硬件依据 | `asc-devkit/docs/zh/api/SIMD-API/c_api/cube_compute/asc_enable_hf32.md` |
