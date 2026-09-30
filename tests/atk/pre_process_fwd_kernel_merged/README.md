# PreProcessFwdKernelMerged ATK 工程

本目录提供 `pre_process_fwd_kernel_merged` 的 ATK 单算子工程：
`executor_pre_process_fwd_kernel_merged.py`、`gen_pre_process_fwd_kernel_merged.py`、
`pre_process_fwd_kernel_merged.yaml`、三份验收 JSON 与 `scripts/`。

## 算子做什么

CP（context parallel）场景下 GDN / KDA（/ DPLR）前向的 **pre-process 融合算子**：
一次调用处理**一个打包窗口**（窗口内可含多段序列，段边界由 `cu_seqlens` 给出），
为窗口内每条链算窗口边界状态 `h` 与仿射链矩阵 `m`，按 `[h | m]` 打包输出 `hm`。
上游的 `all_gather_into_tensor` / `merge_fwd_bwd_kernel` 是编排层的事，不在本算子。

## 输入约束

- 布局 **BNSD `[B, H, T, D]`**（与仓内其它 AscendC 算子一致）；**`B` 恒为 1**（CP 契约：
  `fla/ops/cp/README.md` "CP expects `B == 1` for varlen"）。定长 dense 输入需调用方先行打包。
- `cu_seqlens` **必给**，是 **host `list[int]`**（不是张量），`[N+1]` 严格递增，
  `0 <= cu[0] < cu[-1] <= T`；**允许子区间**（`cu[0] > 0` 或 `cu[-1] < T`）——
  这正是竞品 `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` 的调用形态。
  `Nseq = len(cu_seqlens) - 1`，输出前导维即 `Nseq`。
- **`k` 恒为 `[1, HK, T, K]`**（即使 gk/GVA 路径）；`w`/`u`/`v`/`g`/`gk` 在 `HV` 维。
  `HK` 与 `HV` 必须成倍数（`HV % HK == 0`），`hk = hv // (HV/HK)`；**GVA（`HK < HV`）合法**。
- `gk` 路径（KDA/DPLR）仍是 `gk[1, HV, T, K]`（**按 value head 给门控**），`k` 按 HK 头。
  ⚠ 不要因为"KDA"就把 `k` 建成 `HV` 头 —— `scripts/npu_smoke_ppfm.py::build_case` 与
  `reference/reference.py` 都是"k 按 HK、gate 按 HV"。
- `g` / `gk` **二选一**（互斥）；`g` 是 `[1, HV, T]`、`gk` 是 `[1, HV, T, K]`，
  两者都是 **base-2 的 chunk 内累积对数衰减**，dtype 支持 FP32 / BF16。
- `K = V = 128`、`chunk_size = 64` 为固定规格（host 拦截其它值）；`k/w/u/v` 支持 BF16 / FP16。
- `bg`（DPLR）与 `v != u` 的路径**本轮不验收**（见下方 TilingKey 表）。
- 输出 `hm[Nseq, HV, K, V+K]` **FP32**；`hm[i,hv][:, 0:V]` 是 `h`、`[:, V:V+K]` 是 `m`。

### ⚠ 用例数据的形态要求（不是可选项）

`w` 必须取**模型同构**分布：`k` 逐行归一化、`w = beta · k`（`beta ~ U(0, 0.02)`）。
这与 `scripts/npu_smoke_ppfm.py::build_case` 一致；若改用满幅随机 `w`，
`m = Π M_c` 会把 fp32 求和顺序的 1 ulp 差异放大到 O(1)（`|m| ~ 1e7`）——
那是**用例病态**，不是实现缺陷。本工程的 `executor_*.py` 已经按模型同构构造输入。

## 标杆来源

| 项 | 内容 |
| --- | --- |
| CPU 标杆（唯一真源） | `fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference/reference.py` |
| 上游语义来源 | `fla-org/flash-linear-attention@e52dbc0e` → `fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` |
| 算子自身用例表 | `fla/ops/.../pre_process_fwd_kernel_merged/cases.json`（41 条：37 精度 + 4 性能） |
| 接口与约束 | `fla/ops/.../pre_process_fwd_kernel_merged/docs/api.md` §3 |

`executor_*.py` 按相对路径**加载**上面那份参考实现（`importlib`），不复制副本，
避免出现两份会分叉的标杆；输入构造 / `run_cpu` / `run_npu` / `FunctionApi` 都在本目录。

### 精度口径（重要）

- **验收基线 = 契约版标杆**：`accum_dtype=fp32` + 三个舍入点开关全开
  （`h`/`v_new` 进 MMAD 前降到输入 dtype、`M_c@m` 每 chunk 回落 fp32）。
- `reference.py` 的模块文档写明：kernel 的 h/m 累加器是 **FP32**，用 FP64 基准会让任何
  忠实实现平白多出 **~9.4e-3** 的绝对偏差（与 H20 `ieee` 对齐时实测）。
  ⇒ 本工程的 **`high_precision=True`（ATK benchmark 节点）只作参考侧灵敏度对照**，
  不作为验收真值；验收真值走 `high_precision=False`。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950`，可配合统一脚本的
`-soc=ascend910b|ascend910_93|ascend950` 使用（默认 `auto` 由 `npu-smi` 探测）。
本算子当前验收平台为 **A5 / Ascend950**；A2/A3 侧 910B 已通过编译与门禁。

## 泛化与用例生成

取值空间由 `pre_process_fwd_kernel_merged.yaml` 的 `valid/invalid` 声明；
`gen_pre_process_fwd_kernel_merged.py` 把该空间冻结为**确定性子集**（`frozen_by_gen`），
逐条把完整 `case_spec` 与 attrs 写入 JSON：

| 文件 | 条数 | 来源 |
| --- | ---: | --- |
| `atk_pre_process_fwd_kernel_merged.json` | **37** | 算子 `cases.json` 的 PPFM-01..37（逻辑分支/边界/变长/GVA/gate dtype/并行度/子区间） |
| `atk_pre_process_fwd_kernel_merged_perf.json` | **4** | 用户模型 case（PPFM-38..41：model-g / model-gk / CP=2+GVA / 长窗口） |
| `atk_pre_process_fwd_kernel_merged_mss.json` | **5** | 按**可达 TilingKey** 人工构造（4 个 key + 1 条变长段枚举） |

三条来源不同、不能互相替代。生成（不需要 ATK 环境即可落地 JSON）：

```bash
python gen_pre_process_fwd_kernel_merged.py --summary
# accuracy=37 perf=4 mss=5 tiling_keys=4 -> [('g','bf16'), ('g','fp32'), ('gk','bf16'), ('gk','fp32')]
```

`atk case -f pre_process_fwd_kernel_merged.yaml -p gen_pre_process_fwd_kernel_merged.py -dt 100 -en 0`
等价于走 ATK 的生成入口；生成结果审查后再更新 `atk_*.json`。

## TilingKey 覆盖表

来源：算子 `docs/design.md` §3.2.1「`gate ∈ {USE_G, USE_GK, USE_BG}` × `gate dtype ∈ {BF16, FP32}`
= 6 个 TilingKey」，本轮**验收 4 个**（DPLR 的 2 个已注册、不验收）。

| TilingKey | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `USE_G` + gate **FP32** | 给 `g`（FP32）、`bg` 缺省 | `PPFM-01..05`、`PPFM-11..14`、`PPFM-19..26`、`PPFM-30`、`PPFM-32`、`PPFM-34`、`PPFM-36` | `PPFM-11`（T=1）等 | `MSS-gate-g-fp32` | A2/A3/A5 | **待补**：host tiling UT 或运行时 tilingKey 记录（上电后执行，见 §验收） |
| `USE_G` + gate **BF16** | 给 `g`（BF16）、`bg` 缺省 | `PPFM-27` | `PPFM-27` | `MSS-gate-g-bf16` | A2/A3/A5 | **待补** |
| `USE_GK` + gate **FP32** | 给 `gk`（FP32）、`bg` 缺省 | `PPFM-06..10`、`PPFM-15..18`、`PPFM-28`、`PPFM-31`、`PPFM-33`、`PPFM-35`、`PPFM-37` | `PPFM-06`（T=1）等 | `MSS-gate-gk-fp32`、`MSS-varlen-3seg` | A2/A3/A5 | **待补** |
| `USE_GK` + gate **BF16** | 给 `gk`（BF16）、`bg` 缺省 | `PPFM-29` | `PPFM-29` | `MSS-gate-gk-bf16` | A2/A3/A5 | **待补** |
| `USE_BG`（DPLR）+ FP32 | 给 `bg` + `gk`，`v != u` | — | — | — | — | **本轮不验收**：算子 host 侧尚未开放该路径（`use_bg` 在用例空间里为 `false`） |
| `USE_BG`（DPLR）+ BF16 | 同上 | — | — | — | — | **本轮不验收**，同上 |

> 「实际选择证据」按 `tests/atk/README.md` 的硬要求：**必须补 host tiling UT 或运行时记录**，
> 没有实际选中证据时不得标记为已覆盖。上电后第一步就补这一列。

## 三类映射（硬要求）

1. **逻辑分支 → 精度 case id**

| 逻辑分支 | 精度 case id |
| --- | --- |
| 窗口规模 / 尾块（GDN） | `PPFM-01..05` |
| 窗口规模 / 尾块（KDA） | `PPFM-06..10` |
| 变长多段（GDN） | `PPFM-11..14` |
| 变长多段（KDA） | `PPFM-15..18` |
| GVA（`HK < HV`，1:2 / 1:3 / 2:1 / 3:1 / 4:1 / 8:1 / 32:1） | `PPFM-19..25` |
| gate dtype（g FP32/BF16、gk FP32/BF16） | `PPFM-26..29` |
| 并行度（`Nseq × HV`：8 / 16 / 32 / 64；`Nseq` = 1 / 16 / 64） | `PPFM-30..35` |
| 子区间窗口（`bos > 0`；含 `eos < T`） | `PPFM-36..37` |

2. **用户模型 case → 性能 case id**

| 用户模型 case | 性能 case id |
| --- | --- |
| H20 model-gk（KDA） | `PPFM-38` |
| H20 model-g（GDN） | `PPFM-39` |
| CP=2 窗口 + GVA | `PPFM-40` |
| 长窗口 | `PPFM-41` |

3. **可达 TilingKey → `_mss.json` case id**：见上一节表格。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=pre_process_fwd_kernel_merged -scope=gen_cases
```

运行前按 `tests/atk/README.md`「运行前准备」准备 `ATK_ENV / CANN_ENV / FLA_NPU_ENV`，
并确认 `atk --version` ≥ `26.8.8`。

> **验收前先做「同包自比」**：本算子此前的位级判据发现过"只在 colSplit=1（`cb_=CV_V`）的形状上
> 不一致"的现象。ATK 的 `-scope=determinism`（`accuracy_dc`）
> 正好覆盖这一点，建议在 `all` 之前先单独跑一次 `-scope=determinism`。

## 验收结果记录

> **状态：待上电后执行并回填。** 本工程是"可执行的交付件"；下面的表格按
> `tests/atk/README.md`「验收结果记录」的要求预置，执行后逐格填实测值。

| 项 | 内容 |
| --- | --- |
| CPU 标杆版本 | `reference/reference.py`（SHA256 待回填） |
| 三份测试文件版本 | `atk_*.json` / `_perf.json` / `_mss.json`（见 git 版本） |
| 被测代码版本 | `feat/ppfm-tile-a5` @ 待回填 |
| 目标 SoC | ascend950（A5） |
| 执行的测试动作 | `accuracy` / `performance` / `determinism` / `mssanitizer` |
| 精度用例总数 / 失败数 | 37 / 待回填 |
| 逻辑分支/边界/异常/TilingKey/确定性/内存覆盖结论 | 待回填 |

**性能用例（`_perf.json`）逐 case 结果**：

| case id | 模型 shape | SoC/dtype | 对比基线 | 性能目标 | 实测性能 | 与基线比值 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `PPFM-38` | KDA `T=11264, HK=HV=32` | ascend950/bf16 | H20 3329.4 µs（cp=8） | — | 待测 | — | 待测 |
| `PPFM-39` | GDN `T=11264, HK=HV=32` | ascend950/bf16 | H20 1620.7 µs | — | 待测 | — | 待测 |
| `PPFM-40` | GDN `T=5632, HK=16 HV=32` | ascend950/bf16 | — | — | 待测 | — | 待测 |
| `PPFM-41` | GDN `T=16384, HK=HV=32` | ascend950/bf16 | — | — | 待测 | — | 待测 |

> 参考：开发期在 Ascend950 上用 `msprof op` 实测的 Task Duration 为
> gdn 模型 case ≈1726 µs、kda `T=16384/HK=HV=64` ≈4227 µs。
> ATK 的 performance 阶段用的是它自己的统计口径，回填时请注明统计方式。

## 提交前检查

```bash
rg -n "_ascendc_common_executor|parents\[1\]|parents\[3\]" tests/atk/pre_process_fwd_kernel_merged
rg -n "atk_output|result/|\.xlsx|__pycache__" tests/atk/pre_process_fwd_kernel_merged
python -m py_compile tests/atk/pre_process_fwd_kernel_merged/*.py
```

预期：公共工具只从 `tests/atk/common/` 引入；不提交 `atk_output/`、`result/`、XLSX 与 `__pycache__`。
