## 一、详设文档

| 文档类型 | 链接/路径 |
|---|---|
| 算子说明 | `fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/README.md` |
| GVA + V=256 设计 | `bwd_dhu_GVA_VDIM256.md` |
| 需求背景说明 | 在 GVA（Grouped Value Attention）场景下验证 `chunk_gated_delta_rule_bwd_dhu` 单算子精度：`Hv = group_size * Hk`，输出 `dh`（分块隐藏状态梯度）与 `dv2`（融合后的 Value 梯度）。 |

GVA 能力合入 commit：`ce8573ae`（Hk/Hv 拆分、cube/vec 头映射、OpApi 输出对齐 Hv）。

## 二、接口说明

`chunk_gated_delta_rule_bwd_dhu` 输入为 `q, k, w, d_o, dv, g` 等，输出为 `dh, dh0, dv2`。

| 路径 | Python 接口 | 说明 |
|---|---|---|
| torch_custom / aclnn 路径 | `torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu` | 通过 `aclnnChunkGatedDeltaRuleBwdDhu` 进入 AscendC 算子 |

说明：

- `gK` 当前未启用，须传 `None`。
- `h0` / `dht` 本轮用例均为 `None`（仅测 `dh`、`dv2`）。
- `use_exp2` / `transpose_state_layout` 当前须为默认/未传。
- **GVA 矩阵 / cases.json CPU 路径**：随机输入（`create_bwd_dhu_random_inputs`），无需 example dump。
- **GPU 双标杆路径**：依赖 GPU example dump 的 `.pt` 输入与 GPU 输出作为 bench（`gpu/cases.json` 中 `chunk_size=64` 项）。

## 三、算子约束

### 3.1 数据类型限制

| 参数 | 数据类型 |
|---|---|
| `q, k, w, d_o, dv, dh, dv2` | `BFLOAT16` / `FLOAT16` |
| `g` | `FLOAT`（GVA 矩阵固定 fp32）或 与 `q` 同 dtype |
| `cu_seqlens, chunk_indices` | `INT64` |

TilingKey：`1` = g 与 q 同 dtype；`2` = g 为 fp32（GVA 矩阵走 Key=2）。

### 3.2 Shape 约束

| 张量 | Shape | 约束说明 |
|---|---|---|
| `q` | `[B, Hk, T, K]` | Q/K head 数量为 `Hk` |
| `k` | `[B, Hk, T, K]` | 与 `q` 对齐 |
| `w` | `[B, Hv, T, K]` | 与 value head 对齐 |
| `d_o` | `[B, Hv, T, V]` | 与 `dv` 对齐 |
| `dv` | `[B, Hv, T, V]` | Value 上游梯度 |
| `g` | `[B, Hv, T]` | gate，与 value head 对齐 |
| `dh` | `[B, Hv, chunkNum, K, V]` | 各 chunk 起始隐藏状态梯度 |
| `dv2` | `[B, Hv, T, V]` | 融合隐藏状态贡献后的 dv |

### 3.3 GVA 约束

- 支持 `Hv = group_size * Hk`，且 `Hv % Hk == 0`。
- 每个 value head 映射到 `hq = h // (Hv // Hk)` 的 q/k head。
- cube/vec 按 **GVA 方案 A**：粗 task `B×Hk×seqNum`，组内各 value head 串行处理。

### 3.4 变长模式约束

- `cu_seqlens` 与 `chunk_indices` 须同时提供；均为 `None` 时为定长模式。
- 变长模式当前约束 `B = 1`。

### 3.5 其他约束

- `chunk_size` 仅支持 `64` 或 `128`。
- `K ≤ 128`，`V ≤ 256`。
- GPU dump 路径当前仅可靠支持 `chunk_size = 64`（见 `gdn_case_utils.GPU_DUMP_CHUNK_SIZES`）。

### 3.6 硬件版本

- 本轮验证：**Ascend 910B3**（A2 路径）
- vendor：`fla_npu_transformer`（CANN 8.5.0，2026-07-03 重编安装）

## 四、功能自测

> 整合日期：**2026-07-03**  
> 分支：`feat/recompute-wu-gpu-dump-dual`  
> 用例矩阵参考：`GDN泛化用例表.xlsx`（33 项泛化设计）↔ `fla/ops/ascendc/gdn/cases.json`（42 项可执行用例）  
> CPU-only 名单（8 项）：`gdn_cpu_dual_casesjson.DEFAULT_GPU_UNSUPPORTED_CASES`  
> 结论：**GPU dual 30 项 PASS**；**CPU unsupported batch 8 项**中 7/3 已跑 2 项（1 PASS / 1 FAIL），其余待全量重跑。

### 4.1 环境信息

| 项目 | 值 |
|------|-----|
| CANN 版本 | 8.5.0 |
| NPU 型号 | Ascend910B3 |
| Python 环境 | conda `wnc` |
| NPU 设备 | `TEST_DEVICE_ID=2/4/7`（按空闲卡） |
| vendor | `/data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer` |

环境准备：

```bash
conda activate wnc
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer/bin/set_env.bash
export TEST_DEVICE_ID=4
```

### 4.2 编译与安装

| 阶段 | 结果 | 说明 |
|------|------|------|
| custom op 整包编译（910B） | OK | `bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ccache false` |
| .run 安装 | OK | `build_out/fla-npu-fla_npu_linux-aarch64.run --install-path=.../cann-8.5.0` |
| kernel 完整性 | OK | 安装后 `ascend910b/chunk_gated_delta_rule_bwd_dhu/` 存在 |

### 4.3 测试脚本

| 脚本 | 双标杆类型 | 用途 |
|------|-----------|------|
| `fla/.../test/run_bwd_dhu_gpu_dump_dual.sh` | **GPU dual** | cases.json 中 `chunk_size=64` 且 GPU dump 可采集的 case |
| `fla/.../test/test_bwd_dhu_gpu_dump_dual.py` | **GPU dual** | 读取 GPU dump `.pt`，`ct.dual(npu, fp64, gpu_bench)` |
| `torch_custom/fla_npu/test/run_bwd_dhu_gva_cases.sh` | **CPU dual** | cases.json 中 8 项 CPU-only batch |
| `torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py` | **CPU dual** | 随机输入，`ct.dual(npu, fp64, npu_bench)` |

常用命令：

```bash
# GPU 双标杆（cases.json，需 GPU dump）
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test
./run_bwd_dhu_gpu_dump_dual.sh /data/GPU_DUMP --case phase_1_fix_1

# CPU 双标杆（cases.json 8 项 unsupported batch）
cd torch_custom/fla_npu/test
BWD_HU_SUITE=unsupported TEST_DEVICE_ID=4 bash run_bwd_dhu_gva_cases.sh

# 单 case
BWD_HU_CASE=gva_fix_3 TEST_DEVICE_ID=4 bash run_bwd_dhu_gva_cases.sh
```

### 4.4 测试日志

| 轮次 | 日志路径 | 说明 |
|------|----------|------|
| GVA 整合 | `torch_custom/fla_npu/test/bwd_dhu_gva_aclnn_consolidated.md` | 6/25 最终结论 |
| GVA 完整轮 | `torch_custom/fla_npu/test/bwd_dhu_gva_aclnn_20260625_v5.log` | NPU2，10 PASS + 1 SKIP |
| vendor 重编 smoke | NPU4 2026-07-03 | `smoke_varlen_t256_v256` dh/dv2 PASS |
| unsupported batch（旧 9 项） | `fla/ops/ascendc/gdn/dual_benchmark_logs/bwd_dhu/archive_20260703/bwd_dhu_unsupported_20260703_npu4.log` | 部分完成；gva_fix_3 dv2 FAIL |

## 五、精度测试

### 5.1 双标杆策略

| 类型 | 标杆组合 | 输入来源 | 脚本 |
|------|---------|---------|------|
| **CPU dual** | `ct.dual(npu, fp64_gt, npu_bench)` | 随机生成 | `test_npu_bwd_dhu_gva.py` |
| **GPU dual** | `ct.dual(npu, fp64_gt, gpu_bench)` | GPU example dump | `test_bwd_dhu_gpu_dump_dual.py` |

| 档位 | `golden_mode` / bench | 语义 | dual 角色 |
|------|----------------------|------|-----------|
| 升精度真值 | `fp64` | fp64 累加 | expect（gt） |
| NPU 同精度 | `npu` | bf16/fp16 乘 + fp32 累加 | **CPU dual 的 bench** |
| GPU 输出 | GPU dump tensor | Triton/CUDA 路径 | **GPU dual 的 bench** |

精度通过标准（L1）：`MARE_ratio ≤ 5.0`、`MERE_ratio ≤ 1.5`、`RMSE_ratio ≤ 1.5`、`ERR_COUNT_ratio ≤ 2.0`；**`dh` 与 `dv2` 均须 PASS**。

### 5.2 全量用例汇总（cases.json × 泛化用例表）

> 表头：**第 1 列 = cases.json 名称**；**第 2 列 = Excel「新用例 ID」**（无对应设计行则留空）。  
> Shape 列取 cases.json 实际执行参数（与 Excel 设计稿有少量差异时在备注说明）。  
> 不含 `test_npu_bwd_dhu_gva.py` 内置 GVA 矩阵（`smoke_*` / `varlen_t*` 等）。

| 分组 | 双标杆 | 数量 | PASS | 脚本 |
|------|--------|-----:|-----:|------|
| A. GPU 可采集（`chunk_size=64`，非 CPU-only） | GPU dual | **30** | **30** | `test_bwd_dhu_gpu_dump_dual.py` |
| B. CPU-only batch（`DEFAULT_GPU_UNSUPPORTED_CASES`） | CPU dual | **8** | **1**（7/3 部分跑） | `test_npu_bwd_dhu_gva.py` + `BWD_HU_SUITE=unsupported` |
| C. legacy smoke（`chunk_size=128`，无 Excel 行） | CPU dual | **4** | **4** | 同上 / GPU dump 不可用 |
| **合计** | — | **42** | **35+待测** | `fla/ops/ascendc/gdn/cases.json` 全量 |

**Excel 未单独落 cases.json 的设计行**（本轮无对应执行项）：`BSND_noGVA_V128_14`~`16`（C14~C16）、`BSND_GVA_V256_22`（与 21 参数同）、`BSND_GVA_V256_32`~`33`（H=48 未入 cases.json）。

**7/3 CPU batch 快照**（旧 9 项脚本，含已移除的 fix_11/12）：

| cases.json | 结果 | 说明 |
|---|---|---|
| `gva_fix_3` | **FAIL** | dh PASS；dv2 `ERR_COUNT_ratio` ~9.99（small_err FAIL） |
| `gva_var_5` | **PASS** | |
| `gva_var_2/3/6` | SKIP（旧） | 旧等价 SKIP 已移除；现须正式跑 |
| `phase_1_fix_11/12` | PASS（旧 batch） | 已移出 CPU-only，改走 GPU dual |

---

### 5.3 A. GPU 双标杆（30 项）

> 条件：`chunk_size=64` 且不在 `DEFAULT_GPU_UNSUPPORTED_CASES`（8 项）。比对接口 `ct.dual(npu, fp64_gt, gpu_bench)`。

| cases.json | Excel 新用例 ID | 模式 | B | V_H | K_H | T | Vdim | Kdim | chunk_size | cu_len | 双标杆 | 结果 | 描述/备注 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|:---:|---|
| `fix_hk_eq_hv_1` | | 定长 | 1 | 2 | 2 | 128 | 128 | 128 | 64 | - | GPU dual | **PASS** | legacy smoke HK==HV |
| `fix_hk_eq_hv_3` | | 定长 | 4 | 8 | 8 | 512 | 256 | 128 | 64 | - | GPU dual | **PASS** | legacy smoke V=256 |
| `fix_hk_eq_hv_5` | | 定长 | 1 | 32 | 32 | 2048 | 128 | 128 | 64 | - | GPU dual | **PASS** | legacy smoke 长序列 |
| `var_hk_eq_hv_1` | | 变长 | 1 | 4 | 4 | 128 | 128 | 128 | 64 | 2 | GPU dual | **PASS** | legacy 变长 smoke |
| `var_hk_eq_hv_3` | | 变长 | 1 | 16 | 16 | 512 | 256 | 128 | 64 | 4 | GPU dual | **PASS** | legacy 变长 V=256 |
| `var_hk_eq_hv_5` | | 变长 | 1 | 32 | 32 | 2048 | 128 | 128 | 64 | 16 | GPU dual | **PASS** | legacy 变长 长序列 |
| `gva_fix_1` | `BSND_GVA_V256_28` | 定长 | 1 | 32 | 16 | 4096 | 256 | 128 | 64 | - | GPU dual | **PASS** | 定长序列 |
| `gva_fix_2` | `BSND_GVA_V256_29` | 定长 | 16 | 63 | 21 | 2048 | 256 | 128 | 64 | - | GPU dual | **PASS** | 定长序列 |
| `gva_fix_4` | `BSND_GVA_V256_31` | 定长 | 176 | 64 | 2 | 24 | 256 | 128 | 64 | - | GPU dual | **PASS** | 定长序列 |
| `gva_var_1` | `BSND_GVA_V256_21` | 变长 | 1 | 32 | 16 | 16384 | 256 | 128 | 64 | 128 | GPU dual | **PASS** | 变长 cu=128 |
| `gva_var_4` | `BSND_GVA_V128_25` | 变长 | 1 | 32 | 16 | 65536 | 128 | 128 | 64 | 668 | GPU dual | **PASS** | Excel T=36621，cases.json T=65536 |
| `phase_1_fix_1` | `BSND_noGVA_V128_01` | 定长 | 64 | 8 | 8 | 1024 | 128 | 128 | 64 | - | GPU dual | **PASS** | 大batch中等序列 |
| `phase_1_fix_2` | `BSND_noGVA_V128_02` | 定长 | 32 | 16 | 16 | 2048 | 128 | 128 | 64 | - | GPU dual | **PASS** | 中batch中长序列 |
| `phase_1_fix_3` | `BSND_noGVA_V128_03` | 定长 | 16 | 32 | 32 | 4096 | 128 | 128 | 64 | - | GPU dual | **PASS** | 小batch长序列 |
| `phase_1_fix_4` | `BSND_noGVA_V128_04` | 定长 | 8 | 32 | 32 | 8192 | 128 | 128 | 64 | - | GPU dual | **PASS** | 极小batch超长序列 |
| `phase_1_fix_5` | `BSND_noGVA_V128_05` | 定长 | 128 | 4 | 4 | 1024 | 128 | 128 | 64 | - | GPU dual | **PASS** | 极限batch，最小head数 |
| `phase_1_fix_6` | `BSND_noGVA_V128_09` | 定长 | 64 | 8 | 8 | 2048 | 128 | 128 | 64 | - | GPU dual | **PASS** | Excel cs=128，cases.json cs=64 |
| `phase_1_fix_7` | `BSND_noGVA_V128_10` | 定长 | 32 | 16 | 16 | 4096 | 128 | 128 | 64 | - | GPU dual | **PASS** | Excel cs=128，cases.json cs=64 |
| `phase_1_fix_8` | `BSND_noGVA_V128_11` | 定长 | 16 | 32 | 32 | 8192 | 128 | 128 | 64 | - | GPU dual | **PASS** | Excel cs=128，cases.json cs=64 |
| `phase_1_fix_9` | `BSND_noGVA_V128_06` | 定长 | 64 | 8 | 8 | 4096 | 128 | 128 | 64 | - | GPU dual | **PASS** | 大batch+长序列组合 |
| `phase_1_fix_10` | `BSND_noGVA_V128_07` | 定长 | 32 | 16 | 16 | 8192 | 128 | 128 | 64 | - | GPU dual | **PASS** | 中batch+超长序列 |
| `phase_1_fix_11` | `BSND_noGVA_V128_08` | 定长 | 16 | 32 | 32 | 16384 | 128 | 128 | 64 | - | GPU dual | **PASS** | 小batch+极长序列(16K) |
| `phase_1_fix_12` | `BSND_noGVA_V128_12` | 定长 | 8 | 32 | 32 | 32768 | 128 | 128 | 64 | - | GPU dual | **PASS** | Excel cs=128，cases.json cs=64 |
| `phase_1_fix_13` | `BSND_noGVA_V128_13` | 定长 | 1 | 32 | 32 | 32768 | 128 | 128 | 64 | - | GPU dual | **PASS** | batch=1 极限长度 T=32K |
| `phase_1_fix_14` | `BSND_noGVA_V128_18` | 定长 | 1 | 16 | 16 | 65536 | 128 | 128 | 64 | - | GPU dual | **PASS** | 定长极限长度 T=64K |
| `phase_1_fix_15` | `BSND_noGVA_V128_18` | 定长 | 1 | 8 | 8 | 131072 | 128 | 128 | 64 | - | GPU dual | **PASS** | 定长极限长度 T=128K |
| `phase_1_fix_16` | `BSND_noGVA_V128_16` | 定长 | 1 | 4 | 4 | 262144 | 128 | 128 | 64 | - | GPU dual | **PASS** | 定长极限长度 T=256K |
| `phase_1_var_1` | `BSND_noGVA_V128_17` | 变长 | 1 | 8 | 8 | 1024 | 128 | 128 | 64 | 65 | GPU dual | **PASS** | Excel cu≈32，cases.json mean_len=65 |
| `phase_1_var_2` | `BSND_noGVA_V128_18` | 变长 | 1 | 16 | 16 | 2048 | 128 | 128 | 64 | 33 | GPU dual | **PASS** | Excel cu≈64，cases.json mean_len=33 |
| `phase_1_var_3` | `BSND_noGVA_V128_19` | 变长 | 1 | 32 | 32 | 4096 | 128 | 128 | 64 | 17 | GPU dual | **PASS** | Excel cu≈16，cases.json mean_len=17 |

---

### 5.4 B. CPU 双标杆 — unsupported batch（8 项）

> 名单 = `gdn_cpu_dual_casesjson.DEFAULT_GPU_UNSUPPORTED_CASES` = `test_npu_bwd_dhu_gva.CASES_JSON_UNSUPPORTED_NAMES`。  
> 比对接口 `ct.dual(npu, fp64_gt, npu_bench)`，随机输入。

| cases.json | Excel 新用例 ID | 模式 | B | V_H | K_H | T | Vdim | Kdim | chunk_size | cu_len | 双标杆 | 结果 | 描述/备注 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|:---:|---|
| `gva_fix_3` | `BSND_GVA_V128_30` | 定长 | 711 | 32 | 4 | 196 | 128 | 128 | 128 | - | CPU dual | **FAIL** | dh PASS；dv2 small_err ~10x（7/3 NPU4） |
| `gva_var_2` | `BSND_GVA_V256_23` | 变长 | 1 | 63 | 21 | 16384 | 256 | 128 | 64 | 2 | CPU dual | 待测 | GVA 变长 cu=2 |
| `gva_var_3` | `BSND_GVA_V256_24` | 变长 | 1 | 32 | 8 | 65536 | 256 | 128 | 128 | 172 | CPU dual | 待测 | GVA 变长 cu=172 |
| `gva_var_5` | `BSND_GVA_V128_26` | 变长 | 1 | 32 | 4 | 65536 | 128 | 128 | 128 | 17 | CPU dual | **PASS** | 变长 cu=17 cs=128（7/3） |
| `gva_var_6` | `BSND_GVA_V256_27` | 变长 | 1 | 64 | 2 | 262144 | 256 | 128 | 64 | 32 | CPU dual | 待测 | T=262144，CPU golden 极慢 |
| `phase_1_var_4` | `BSND_noGVA_V128_20` | 变长 | 1 | 32 | 32 | 8192 | 128 | 128 | 64 | 9 | CPU dual | 待测 | Excel cu≈8，cases.json mean_len=9 |
| `phase_1_var_5` | `BSND_noGVA_V128_17` | 变长 | 1 | 16 | 16 | 32768 | 128 | 128 | 128 | 2 | CPU dual | 待测 | 变长 cs=128 T=32K（对应 Excel V1） |
| `phase_1_var_6` | `BSND_noGVA_V128_18` | 变长 | 1 | 8 | 8 | 65536 | 128 | 128 | 128 | 2 | CPU dual | 待测 | 变长 cs=128 T=64K（对应 Excel V2） |

---

### 5.5 C. CPU 双标杆 — legacy smoke（4 项，无 Excel 行）

| cases.json | Excel 新用例 ID | 模式 | B | V_H | K_H | T | Vdim | Kdim | chunk_size | cu_len | 双标杆 | 结果 | 描述/备注 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|:---:|---|
| `fix_hk_eq_hv_2` | | 定长 | 2 | 4 | 4 | 256 | 128 | 128 | 128 | - | CPU dual | **PASS** | legacy smoke cs=128 |
| `fix_hk_eq_hv_4` | | 定长 | 8 | 16 | 16 | 1024 | 256 | 128 | 128 | - | CPU dual | **PASS** | legacy smoke cs=128 V=256 |
| `var_hk_eq_hv_2` | | 变长 | 1 | 8 | 8 | 256 | 128 | 128 | 128 | 3 | CPU dual | **PASS** | legacy 变长 cs=128 |
| `var_hk_eq_hv_4` | | 变长 | 1 | 32 | 32 | 1024 | 256 | 128 | 128 | 5 | CPU dual | **PASS** | legacy 变长 cs=128 V=256 |

---

### 5.6 典型 dual 指标（抽样）

| cases.json | Excel 新用例 ID | 双标杆 | 输出 | MARE_ratio | MERE_ratio | RMSE_ratio |
|---|---|--------|------|------------|------------|------------|
| `phase_1_fix_1` | BSND_noGVA_V128_01 | GPU | dh / dv2 | ~1.0 / ~1.0 | ~1.0 / ~1.0 | ~0.05 / ~0.03 |
| `gva_fix_1` | BSND_GVA_V256_28 | GPU | dh / dv2 | ~1.1 / ~1.0 | ~1.1 / ~0.1 | ~0.06 / ~0.01 |
| `gva_var_1` | BSND_GVA_V256_21 | GPU | dh / dv2 | ~1.9 / ~1.0 | ~1.3 / ~0.1 | ~0.01 / ~0.01 |
| `gva_fix_3` | BSND_GVA_V128_30 | CPU | dh / dv2 | ~1.8 / ~1.0 | ~1.3 / ~0.1 | ~0.01 / ~0.01 |
| `gva_var_5` | BSND_GVA_V128_26 | CPU | dh / dv2 | ~1.0 / ~1.3 | ~1.1 / ~1.2 | ~0.08 / ~0.03 |

### 5.7 与 fwd_h 测试策略对比

| 项 | fwd_h | bwd_dhu（本轮） |
|---|---|---|
| 用例来源 | example dump / 随机 | **`fla/ops/ascendc/gdn/cases.json`（42 项）** |
| CPU-only 名单 | 同 8 项 | **`DEFAULT_GPU_UNSUPPORTED_CASES`（8 项）** |
| 泛化表 | — | **`GDN泛化用例表.xlsx` ↔ cases.json** |
| Vdim | 128 / 256 | **128 / 256** |
| 标杆档位 | fp64 + npu + fp32 / GPU | **fp64 + npu（CPU）** / **fp64 + gpu（GPU）** |
| g dtype | fp32 | **fp32**（GVA TilingKey=2） |
| 输出张量 | `h`, `v_new` | **`dh`, `dv2`** |
| 全量 case 数 | cases.json 矩阵 | **42**（30 GPU + 8 CPU batch + 4 legacy） |

## 六、性能测试

不涉及

## 七、遗留问题与风险

| 项 | 说明 |
|---|---|
| vendor 完整性 | 须用 `vendor_name=fla_npu` 整包编译安装，确保 `bwd_dhu` kernel 在 vendor 内 |
| CPU golden 耗时 | `B` 较大或 `T≥16384` 时 fp64/npu 双档 CPU 标杆慢，建议后台跑 |
| `gva_fix_3` dv2 FAIL | 7/3 NPU4：`ERR_COUNT_ratio≈9.99`，小值域精度退化，待定位 |
| CPU batch 待全量 | 8 项名单更新后须重跑（旧 batch 含 fix_11/12，且 gva_var_2/3/6 曾被 SKIP） |
| NPU 选卡 | 个别物理卡可能 `507033`，需 `npu-smi` 换空闲卡 |
| TilingKey 双编译 | `g=fp32` 须 kernel 双 entry，否则 aclnn 报 `BinaryGetFunctionByEntry` |

## 八、代码与分支信息

| 项 | 值 |
|---|---|
| 分支 | `feat/recompute-wu-gpu-dump-dual` |
| GVA 合入 | `ce8573ae` |
| 设计文档 | `bwd_dhu_GVA_VDIM256.md` |
| GVA 用例 | `torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py` |
| CPU 标杆 | `torch_custom/fla_npu/test/test_bwd_dhu.py` |
| CPU-only 名单 | `fla/ops/ascendc/gdn/gdn_cpu_dual_casesjson.py` |
| GPU 双标杆 | `fla/.../test/test_bwd_dhu_gpu_dump_dual.py` |
| 运行脚本 | `run_bwd_dhu_gva_cases.sh` / `run_bwd_dhu_gpu_dump_dual.sh` |
| cases.json | `fla/ops/ascendc/gdn/cases.json` |
| 泛化用例表 | `torch_custom/fla_npu/test/GDN泛化用例表.xlsx` |
| 测试指南 | `BWD_DHU_TEST.md` |
