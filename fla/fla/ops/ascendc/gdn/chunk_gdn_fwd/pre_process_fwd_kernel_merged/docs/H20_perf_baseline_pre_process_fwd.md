# pre_process_fwd_kernel_merged — H20 性能基线（CP 分档，用于对标）

> 基线用途：本仓 Ascend（950）实现后续的性能验收，以本文的 H20 数据为**1.0× 基准**。
> 数据来源：上游 fla **0.5.2** 的 CP kernel 在 **NVIDIA H20** 上的实测（脚本 `bench_cp_chunk_delta_h_v052.py`）。
> 记录日期：2026-09-28。

---

## 0. 一句话口径（对标前必读）

表里的每一行 = **一个 rank 上、一次 `pre_process_fwd_kernel_merged` 调用**的 device 时间：

| 口径项 | 取值 |
|---|---|
| 被测对象 | 上游 `fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged`（**merged 一次算 h 与 m**） |
| fla 版本 | **0.5.2**（**没有 `AFFINE_CHAIN_PRECISION`/`input_precision`**，也没有 zigzag 布局） |
| 精度档 | NVIDIA 默认 **TF32**（≈ 我们 H20 侧的 `prec=default`） |
| dtype | 输入 `bfloat16`（`k/w/u/v`；`g` 为 fp32；KDA 档的 `gk` 为 **bf16**） |
| 布局 | `[1, T, H, D]`（token-major；我们是 BNSD `[1,H,T,D]`，仅影响对拍时的 permute） |
| 窗口 | **单窗口**：`cu_seqlens=[0, T_rank]`、`MULTI_SEQS=False`、`T=T_rank`（每 rank 的本地分片） |
| 布局（CP） | **contiguous**（0.5.2 无 zigzag：`part_len = T_global // world_size`） |
| 是否含通信 | **不含**。`all_gather`/`merge` 是另一个 kernel（脚本里单独测，`chain=cp-1`） |
| 计时 | `torch.cuda.Event`（同 stream），**每轮一次 launch + synchronize**；warmup=5、iters=20，取 mean/median/min/max |
| 其他 | `hm` 在循环外预分配并复用；`grid=(cdiv(V,64)+cdiv(K,64), HV)=(4,HV)`、`BLOCK_SIZE=64`、`BK1=128` |

**结论：这份基线是"单次调用时间 + 纯 kernel + merged + TF32"口径**，
与我们 Ascend 侧**上板 `msprof op` 的 `Task Duration`** 口径一致；
**不能**与我们 `npu_bench_ppfm.py` 的"背靠背吞吐"数字直接比。

### 4 个档位（profile）

| case | HK(=H) | HV | K/V | T_global | gate | 备注 |
|---|---|---|---|---|---|---|
| `gdn_gva_h16_hv32` | 16 | 32 | 128 | 131072 | `g` | GDN+GVA 主流档 |
| `gdn_h8_hv8` | 8 | 8 | 128 | 131072 | `g` | 小头数档（并行度最紧张） |
| `gdn_long_t256k` | 16 | 32 | 128 | **262144** | `g` | 长序列档（与上档同 HV，验证斜率） |
| `kda_h64_hv64` | 64 | 64 | 128 | 131072 | `gk` | KDA 主流档（逐 K gate） |

---

## 1. 原始基线数据（ms，per rank per call）

| case | cp | T_rank | mean | median | min | max |
|---|---|---|---|---|---|---|
| gdn_gva_h16_hv32 | 2 | 65536 | 8.4746 | 8.4708 | 8.3795 | 8.5714 |
| | 4 | 32768 | 4.2986 | 4.3012 | 4.2548 | 4.3426 |
| | 8 | 16384 | 2.2080 | 2.2074 | 2.1888 | 2.2404 |
| | 16 | 8192 | 1.1757 | 1.1759 | 1.1676 | 1.1893 |
| | 64 | 2048 | 0.3693 | 0.3690 | 0.3634 | 0.3842 |
| gdn_h8_hv8 | 2 | 65536 | 5.1064 | 5.0924 | 5.0637 | 5.1810 |
| | 4 | 32768 | 2.5938 | 2.5928 | 2.5872 | 2.6067 |
| | 8 | 16384 | 1.3541 | 1.3531 | 1.3493 | 1.3650 |
| | 16 | 8192 | 0.7343 | 0.7327 | 0.7298 | 0.7474 |
| | 64 | 2048 | 0.2749 | 0.2702 | 0.2676 | 0.3086 |
| gdn_long_t256k | 2 | 131072 | 16.8392 | 16.8483 | 16.6431 | 17.0367 |
| | 4 | 65536 | 8.4741 | 8.4740 | 8.3816 | 8.5502 |
| | 8 | 32768 | 4.2944 | 4.2905 | 4.2471 | 4.3519 |
| | 16 | 16384 | 2.2088 | 2.2108 | 2.1870 | 2.2314 |
| | 64 | 4096 | 0.6270 | 0.6272 | 0.6214 | 0.6383 |
| kda_h64_hv64 | 2 | 65536 | 12.9540 | 12.9613 | 12.8991 | 13.0060 |
| | 4 | 32768 | 6.5334 | 6.5361 | 6.5057 | 6.5582 |
| | 8 | 16384 | 3.3294 | 3.3294 | 3.3105 | 3.3714 |
| | 16 | 8192 | 1.7287 | 1.7271 | 1.7168 | 1.7615 |
| | 64 | 2048 | 0.5206 | 0.5198 | 0.5166 | 0.5314 |

---

## 2. 派生指标（用 median 拟合 `t = a + b·T_rank`）

| case | 斜率 b | 固定开销 a | 每 chunk(64 token) | 每 (chunk×head) | 说明 |
|---|---|---|---|---|---|
| gdn_gva_h16_hv32 | **127.6 ns/token** | **≈108 µs** | 8.17 µs | **0.255 µs** | HV=32 |
| gdn_h8_hv8 | 75.9 ns/token | ≈115 µs | 4.86 µs | **0.608 µs** | HV=8：并行度不足，单 head 成本 2.4× |
| gdn_long_t256k | 127.8 ns/token | ≈104 µs | 8.18 µs | 0.256 µs | 与 gva 同斜率 → 长序列线性 |
| kda_h64_hv64 | 196.0 ns/token | ≈118 µs | 12.54 µs | **0.196 µs** | HV=64：单 head 成本最低 |

**强扩展效率**（以 cp=2 为基准，理想=按 cp 对折；gdn_gva_h16_hv32）：

| cp | 实测 median | 理想 | 效率 |
|---|---|---|---|
| 2 | 8.4708 ms | — | — |
| 4 | 4.3012 ms | 4.2354 | 98.5% |
| 8 | 2.2074 ms | 2.1177 | 95.9% |
| 16 | 1.1759 ms | 1.0589 | 90.1% |
| 64 | 0.3690 ms | 0.2647 | 71.7% |

→ cp 越大，**固定开销（≈110 µs/次）占比越高**；cp=64 时已占 30%。

---

## 3. 与我们自己的 H20 单点实测交叉校验（同级可信度）

我们此前在 H20 上用 `benchmarks/cp/bench_pre_process_h20.py` 测过同一算子
（**model-gk：T=11264、HK=HV=32、K=V=128、BT=64、`prec=default`、both halves**）：

```
p50 = 1.6207 ms   p90 = 1.6285 ms   min = 1.6066 ms
```

用本文基线（gva 档：`a=0.1077 ms`、`b=1.276e-4 ms/token`）反推 T=11264：
`0.1077 + 11264×1.276e-4 = 1.545 ms` → 与实测 1.6207 ms **相差 +4.9%**（同一口径，可信）。

**因此本文基线与我们的 H20 单点实测是同一测量体系，可以直接作为验收基准。**

---

## 4. 怎么用它给我们打分（对标方法）

1. **对齐口径**：同 case（HK/HV/K/V/T_global/BT/dtype/gate）、同 cp、**只比 fwd**、比
   "**单次调用时间**" —— Ascend 侧取上板 `msprof op` 的 `OpBasicInfo.Task Duration(us)`（或等价的单次 event 时间）。
2. **打分口径**（两档一起看，缺一不可）：
   * 端到端：`ratio = Ascend_ms / H20_ms`（per rank per call）→ 目标 **≤1.0×**；
   * 分解：`固定开销 a` 与 `每 (chunk×head) 成本` 分别对比 —— 只报总数会掩盖"prologue 太贵"或"每 chunk 太慢"。
3. **不允许的比法**：背靠背吞吐 vs 单次 event；`half=h/half=m` 拆分 vs merged；含 `all_gather/merge` vs 纯 kernel；
   不同 `T_global` 或不同 HV 直接比。
4. **报告模板**（建议）：每个 case×cp 一行
   `case | cp | T_rank | H20 median(ms) | Ascend TaskDur(ms) | ratio | a(µs) | 每(chunk×head)(µs) | 备注`

---

## 5. 当前差距快照（Ascend 950 侧，2026-09-28）

Ascend 上板实测（246，GDN，HV=8，TF32 等价口径）得到的成本模型：
**每 block（=1 条 head 链）≈ 98 µs 固定 + 24.5 µs/chunk**（T=64→4096 四点拟合吻合）。

| 档 | H20 基线 | Ascend 现状 | 差距 |
|---|---|---|---|
| `gdn_h8_hv8`，cp=2（T_rank=65536） | **5.106 ms** | 98 µs + 1024×24.5 µs ≈ **25.2 ms**（外推） | **≈4.9×** |
| 每 (chunk×head) 墙钟 | 0.608 µs | 24.5/8 = **3.06 µs** | **≈5.0×** |
| 固定开销 a | ≈115 µs | ≈98 µs | **≈0.85×（我们不输）** |

> ⚠️ 上表 Ascend 一列是**外推**（按已验证的成本模型线性推到 T_rank=65536）；等按 §4 采完同 case 的真实数据再替换。
> 结论方向：**固定开销我们不落后，差距全在"每 chunk 成本"与并行资源数**（H20 一个 rank 用整卡 78 SM×多 warp；
> 我们受 `Nwork=Nseq×HV` 限制，HV=8 时只占 8 个 block）。

### 待补齐的基线（当前没有 / 需补测）

| 项 | 现状 | 需要时怎么做 |
|---|---|---|
| **zigzag** 档 | 0.5.2 无 zigzag → 基线里没有 | 用支持 zigzag 的 fla 版本重跑同一脚本（新增 `--layouts zigzag`） |
| `tf32x3` / `ieee` 档 | 0.5.2 无该参数 | 换新版本 fla 或改用 `input_precision` 重测 |
| `bwd` / `merge` 行 | 脚本测了（`chain=cp-1`），本次只给了 fwd | 把脚本产出的 JSON/CSV 发我，我并入本文 |
| 非整除 / 多序列（Nseq>1）档 | 基线只有单段整窗 | 若模型侧会打包多段，需要补 Nseq>1 的档位 |

---

## 6. 复现命令

**H20 侧（基线）**
```bash
# fla 0.5.2 环境
python bench_cp_chunk_delta_h_v052.py --iters 20 --warmup 5 \
    --cp-sizes 2 4 8 16 64 --layouts contiguous \
    --profiles gdn_gva_h16_hv32 gdn_h8_hv8 kda_h64_hv64 gdn_long_t256k \
    --dtype bfloat16 --save-dir ./out
# 只关心 fwd 时，从输出里取 pre_process_fwd_kernel_merged 的行
```

**Ascend 侧（对标）**
```bash
# 上板（推荐，口径与基线一致）：Task Duration 即"单次调用时间"
PPFM_SIM_T=<T_rank> PPFM_SIM_HV=<HV> PPFM_SIM_HK=<HK> PPFM_SIM_VARIANT=<gdn|kda> \
  msprof op --application="python3 sim_ppfm.py" --output=<out> \
  --kernel-name=PreProcessFwdKernelMerged --launch-count=1 --warm-up=1 --aic-metrics=Default
# 汇总分析（脚本见交付目录）：python3 analyze_msopprof.py T:HV:<OPPROF目录>
```
> 注意：Ascend 侧窗口按 BNSD + 子区间传入即可与基线同 case 对齐（`cu_seqlens=[0,T_rank]` 或子区间都支持）。

---

## 7. 文件与出处

| 项 | 位置 |
|---|---|
| 基线脚本 | `bench_cp_chunk_delta_h_v052.py`（fla 0.5.2 CP kernel 基准，用户提供） |
| 本文档 | 本地 `outputs/profiling/H20_perf_baseline_pre_process_fwd.md`；246 `ppfm-sim/` 与仓 `…/pre_process_fwd_kernel_merged/docs/` |
| 我们的 H20 单点 | `benchmarks/cp/bench_pre_process_h20.py`（`--case model-gk --precision default`） |
| Ascend 采集/分析 | `sim_ppfm.py`、`analyze_msopprof.py`（见 `PROFILING_GUIDE.md`） |
| 接口/调用形态对齐 | `docs/interface_vs_competitor.md` |
