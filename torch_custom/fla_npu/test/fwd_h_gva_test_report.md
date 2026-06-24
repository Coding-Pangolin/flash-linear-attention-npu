# chunk_gated_delta_rule_fwd_h GVA 精度测试报告

> 分支：`feat/fwd-h-gva-test`  
> 测试日期：2026-06-23  
> 设备：Ascend 910B3，NPU 7  
> CANN：8.5.0 + vendor `fla_npu_transformer_transformer`

## 1. 概述

本报告覆盖 `chunk_gated_delta_rule_fwd_h` 算子在 GVA（Grouped Value Attention）场景下的精度验证流程与结果。

测试采用 **三档标杆** 策略：

| 档位 | `golden_mode` | 语义 | 用途 |
|------|---------------|------|------|
| 升精度真值 | `fp64` | fp64 累加 | `ct.dual` 的 expect（gt） |
| **NPU 同精度** | **`npu`** | **bf16 乘 + fp32 累加，状态 bf16 回写** | **`ct.dual` 的 bench（主标杆）** |
| 旧同精度 | `fp32` | fp32×fp32 矩阵乘 + fp32 累加 | 仅存档对比 |

NPU 实际计算路径：`k/w/u` 以 bf16 读入 GM → Cube **bf16×bf16** 矩阵乘、**fp32 workspace 累加** → vector 侧 float 运算 → 输出 cast bf16。  
因此 dual 的 bench 必须使用 `golden_mode="npu"`，否则长序列会因 fp32 全精度乘的额外舍入差异被误判为算子 bug。

比对接口：`ct.dual(npu_out, ref_fp64, ref_npu)`，level=L1。

---

## 2. 脚本与用法

### 2.1 环境准备

```bash
source /data/zs/run/8.5/ascend-toolkit/set_env.sh
source /data/zs/run/8.5/cann-8.5.0/vendors/fla_npu_transformer_transformer/bin/set_env.bash
conda activate wnc

export TEST_DEVICE_ID=7   # 推荐 NPU 7
```

算子需预先编译安装（示例）：

```bash
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_fwd_h --vendor_name=fla_npu_transformer
bash build/fla-npu-fla_npu_transformer_linux-aarch64.run \
  --install-path=/data/zs/run/8.5/cann-8.5.0 --install-for-all --quiet
```

### 2.2 主脚本 `run_fwd_h_gva_cases.sh`

**流程**：`examples/flash_gated_delta_rule.py`（`GDN_FWD_H_DUMP_EXIT=1`）dump 模型分布输入 → `test_npu_fwd_h_gva.py` 单算子 dual 比对 → 输出 tensor + ct.viz 图片。

```bash
cd torch_custom/fla_npu/test

# 全部用例（缺 dump 则自动 dump，已有则复用）
TEST_DEVICE_ID=7 bash run_fwd_h_gva_cases.sh

# 单用例
FWD_H_CASE=fixed_t4096_v128 TEST_DEVICE_ID=7 bash run_fwd_h_gva_cases.sh

# 仅 dump 输入（不跑精度）
FWD_H_DUMP_ONLY=1 FWD_H_CASE=varlen_t65536_v128_cu668 TEST_DEVICE_ID=7 bash run_fwd_h_gva_cases.sh

# 仅测已有 dump（跳过 example，快速回归）
FWD_H_TEST_ONLY=1 TEST_DEVICE_ID=7 bash run_fwd_h_gva_cases.sh

# 强制重新 dump
FWD_H_FORCE_DUMP=1 FWD_H_CASE=smoke_fixed_t4096_v128 TEST_DEVICE_ID=7 bash run_fwd_h_gva_cases.sh
```

**环境变量**

| 变量 | 默认 | 说明 |
|------|------|------|
| `TEST_DEVICE_ID` | 5 | NPU 设备号 |
| `GDN_FWD_H_DUMP_DIR` | `examples/.../chunk_gated_delta_rule_fwd_h/data` | 输入 dump 目录 |
| `FWD_H_OUT_DIR` | `torch_custom/fla_npu/test/fwd_h_out` | 输出 tensor / viz 目录 |
| `FWD_H_CASE` | （空=全部） | 只跑指定用例名 |
| `FWD_H_TEST_ONLY` | 0 | 1=跳过 dump，要求已有 `.pt` |
| `FWD_H_DUMP_ONLY` | 0 | 1=只 dump 不测 |
| `FWD_H_FORCE_DUMP` | 0 | 1=忽略已有 dump 重新生成 |
| `FWD_H_VIZ` | 1 | 1=生成 ct.viz 图片 |
| `FWD_H_VIZ_SAMPLE_COUNT` | 200000 | viz 采样点数 |

**输出目录结构**（每个 case 一个子目录）：

```
fwd_h_out/<case_name>/
├── outputs.pt              # 汇总
├── h_npu.pt / h_ref_fp64.pt / h_ref_npu.pt / h_ref_fp32.pt
├── v_new_npu.pt / v_new_ref_fp64.pt / v_new_ref_npu.pt / v_new_ref_fp32.pt
└── viz/
    ├── <case>_h_npu_vs_fp64_Standard.png
    └── <case>_v_new_npu_vs_fp64_Standard.png
```

### 2.3 补跑脚本 `run_fwd_h_missing_cases.sh`

针对历史上缺少 input dump 的 6 个用例，依次执行 dump + dual（内部强制 `FWD_H_TEST_ONLY=0`）：

```bash
TEST_DEVICE_ID=7 bash run_fwd_h_missing_cases.sh
# 日志默认：/tmp/fwd_h_missing_cases.log
```

### 2.4 手工 dump 示例

```bash
GDN_FWD_H_DUMP_DIR=examples/fast_kernel_launch_example/tests/chunk_gated_delta_rule_fwd_h/data \
GDN_FWD_H_DUMP_NAME=varlen_t16384_v128_cu2 \
GDN_FWD_H_DUMP_EXIT=1 \
python3 examples/flash_gated_delta_rule.py \
  --device 7 --batch 1 --tokens 16384 \
  --query-heads 21 --value-heads 63 \
  --key-dim 128 --value-dim 128 --chunk-size 64 \
  --dtype bf16 --mean-len 16384
```

---

## 3. 用例矩阵

共 12 个用例（Vdim=128），定义见 `test_npu_fwd_h_gva.py` 中 `CASES`。

| 用例名 | B | K_h/V_h | T | cs | varlen | 状态 |
|--------|---|---------|---|----|--------|------|
| varlen_t16384_v128_cu128 | 1 | 16/32 | 16384 | 64 | cu128 | 可测 |
| varlen_t16384_v128_cu2 | 1 | 21/63 | 16384 | 64 | cu2 | 可测 |
| varlen_t65536_v128_cu668 | 1 | 16/32 | 65536 | 64 | cu668 | 可测 |
| varlen_t65536_v128_cu17 | 1 | 4/32 | 65536 | 128 | cu17 | **SKIP** |
| varlen_t65536_v128_cu172 | 1 | 8/32 | 65536 | 128 | cu172 | **SKIP** |
| varlen_t262144_v128_cu32 | 1 | 2/64 | 262144 | 64 | cu32 | **SKIP** |
| fixed_t4096_v128 | 1 | 16/32 | 4096 | 64 | 定长 | 可测 |
| fixed_b16_t2048_v128 | 16 | 21/63 | 2048 | 64 | 定长 | 可测 |
| fixed_b176_t24_v128 | 176 | 2/64 | 24 | 64 | 定长 | 可测 |
| fixed_b711_t196_v128 | 711 | 4/32 | 196 | 128 | 定长 | **SKIP** |
| smoke_fixed_t4096_v128 | 1 | 16/32 | 4096 | 64 | 定长 | 可测 |
| smoke_varlen_t256_v128 | 1 | 16/32 | 256 | 64 | cu5 | 可测 |

SKIP 原因均为 **example 前置 Triton 算子**无法完成 dump，与 fwd_h 单算子本身无关：

- `cs=128` 或 `B=711`：`chunk_scaled_dot_kkt` UB overflow
- `T=262144`：aicore 507015（MTE DDR 地址越界）

---

## 4. 测试结果

### 4.1 第一轮：npu 对齐 dual bench 回归（`FWD_H_TEST_ONLY=1`）

日志：`/tmp/fwd_h_npu_aligned_run.log`，2026-06-23，NPU 7。

| 用例 | h | v_new | 备注 |
|------|---|-------|------|
| varlen_t16384_v128_cu128 | PASS | PASS | 旧 fp32 bench 曾 FAIL |
| fixed_t4096_v128 | PASS | PASS | 旧 fp32 bench 曾 FAIL |
| fixed_b176_t24_v128 | PASS | PASS | |
| smoke_fixed_t4096_v128 | PASS | PASS | 旧 fp32 bench 曾 FAIL |
| smoke_varlen_t256_v128 | PASS | PASS | 旧 fp32 bench 曾 FAIL |
| 其余 6 个 | — | — | 当时缺 dump（见 4.2 补跑） |
| 4 个 SKIP 用例 | SKIP | SKIP | example 前置限制 |

**典型 dual 指标**（`fixed_t4096_v128`）：

| 输出 | MARE_ratio | MERE_ratio | RMSE_ratio |
|------|------------|------------|------------|
| h | 1.0 | ~1.0 | ~1.0 |
| v_new | 1.0 | ~0.15 | ~0.76 |

MARE_ratio≈1 表示 NPU 与 npu 标杆的最大相对误差同量级；MERE/RMSE ratio < 1 表示 NPU 平均误差优于标杆。

### 4.2 第二轮：补 dump + 测试（`run_fwd_h_missing_cases.sh`）

日志：`/tmp/fwd_h_missing_cases.log`，2026-06-23，NPU 7，耗时约 94 分钟。

| 用例 | dump | h | v_new | 说明 |
|------|------|---|-------|------|
| varlen_t16384_v128_cu2 | ✅ 760MB | PASS | PASS | |
| fixed_b16_t2048_v128 | ✅ 1.5GB | PASS | PASS | |
| varlen_t65536_v128_cu668 | ✅ 1.6GB | PASS | PASS | |
| varlen_t65536_v128_cu17 | ❌ | — | — | Triton cs=128 UB overflow |
| varlen_t65536_v128_cu172 | ❌ | — | — | 同上 |
| varlen_t262144_v128_cu32 | ❌ | — | — | aicore 507015 |

### 4.3 汇总（当前可测用例 8/8 PASS）

| # | 用例 | dual 结果 |
|---|------|-----------|
| 1 | varlen_t16384_v128_cu128 | PASS |
| 2 | varlen_t16384_v128_cu2 | PASS |
| 3 | varlen_t65536_v128_cu668 | PASS |
| 4 | fixed_t4096_v128 | PASS |
| 5 | fixed_b16_t2048_v128 | PASS |
| 6 | fixed_b176_t24_v128 | PASS |
| 7 | smoke_fixed_t4096_v128 | PASS |
| 8 | smoke_varlen_t256_v128 | PASS |

**结论**：在 npu 对齐标杆下，所有已成功 dump 的用例 h / v_new dual 均 PASS，长序列误报问题已消除。

---

## 5. 代码变更摘要

| 文件 | 变更 |
|------|------|
| `test_fwd_h.py` | 新增 `golden_mode`（fp32/fp64/npu）；npu 模式 bf16 乘 + fp32 累加 |
| `test_npu_fwd_h_gva.py` | dual 改为 `ct.dual(npu, fp64, npu_bench)`；输出 dump + ct.viz；SKIP 不可 dump 用例 |
| `run_fwd_h_gva_cases.sh` | 增加 `FWD_H_OUT_DIR`/`FWD_H_VIZ`；修正 vendor env 路径 |
| `run_fwd_h_missing_cases.sh` | 补跑 6 个缺 dump 用例的批量脚本 |

---

## 6. 已知限制与后续

1. **4 个 SKIP 用例**需先解决 example 前置 Triton/NPU 问题才能 dump；fwd_h 单算子精度未覆盖这些 shape。
2. **input dump 体积大**（65536 序列约 1.6GB），不建议提交到 git；保留在 `examples/.../data/` 本地目录。
3. **CPU golden 计算慢**：T=65536 / T=16384 的 fp64+npu+fp32 三档标杆在 CPU 上耗时显著，建议已有 dump 时用 `FWD_H_TEST_ONLY=1` 快速回归。
4. **test 输出**（`fwd_h_out/`、viz 图片）为本地产物，不纳入版本库。
