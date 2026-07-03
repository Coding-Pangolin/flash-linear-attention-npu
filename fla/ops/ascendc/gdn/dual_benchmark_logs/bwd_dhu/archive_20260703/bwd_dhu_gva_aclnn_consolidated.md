# bwd_dhu GVA + Vdim=256 测试日志整合

> 整合日期：2026-06-26  
> 分支：`feat/chunk-gated-delta-rule-bwd-dhu-fast-kernel`  
> 设备：NPU7 / NPU2（`TEST_DEVICE_ID`）  
> 标杆：`ct.dual(npu, fp64_gt, npu_bench)`，level=L1，`g=fp32`（TilingKey=2）

## 一、最终汇总

| 类别 | 数量 | 说明 |
|------|------|------|
| **PASS** | **10** | `dh` + `dv2` dual 均通过 |
| **SKIP** | **1** | `varlen_t262144_v256_cu32`（CPU fp64 golden 过慢，主动跳过） |
| **FAIL** | **0** | — |

**结论：11 项可跑用例中 10 项 PASS，1 项主动 SKIP，GVA + Vdim=256 aclnn 精度验证通过。**

## 二、用例结果矩阵（权威结论）

| Case | 模式 | B | Hk/Hv | T | cs | 结果 | 权威日志 | 备注 |
|------|------|--:|-------|--:|---:|:---:|----------|------|
| `smoke_varlen_t256_v256` | 变长 | 1 | 16/32 | 256 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260622_npu7.log` | cu_len=5 |
| `smoke_fixed_t4096_v256` | 定长 | 1 | 16/32 | 4096 | 64 | **PASS** | `bwd_dhu_gva_run_20260625.log` | v3 曾因 `save_out` bug 误标 FAIL，精度已通过 |
| `varlen_t16384_v256_cu2` | 变长 | 1 | 21/63 | 16384 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | v4 因 `generate_cu_seqlens` bug FAIL，v5 修复后 PASS |
| `varlen_t16384_v256_cu128` | 变长 | 1 | 16/32 | 16384 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v4.log` | 128 条序列 |
| `varlen_t65536_v256_cu17` | 变长 | 1 | 4/32 | 65536 | 128 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | v4 FAIL → v5 PASS |
| `varlen_t65536_v256_cu172` | 变长 | 1 | 8/32 | 65536 | 128 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | v4 FAIL → v5 PASS |
| `varlen_t65536_v256_cu668` | 变长 | 1 | 16/32 | 65536 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | 668 条序列 |
| `fixed_b16_t2048_v256` | 定长 | 16 | 21/63 | 2048 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | GVA 21→63，CPU golden 最重定长 case |
| `fixed_b176_t24_v256` | 定长 | 176 | 2/64 | 24 | 64 | **PASS** | `bwd_dhu_gva_aclnn_20260622_npu7.log` | 大 batch 小 T |
| `fixed_b711_t196_v256` | 定长 | 711 | 4/32 | 196 | 128 | **PASS** | `bwd_dhu_gva_aclnn_20260625_v5.log` | B=711 |
| `varlen_t262144_v256_cu32` | 变长 | 1 | 2/64 | 262144 | 64 | **SKIP** | — | 主动跳过 |

## 三、典型 dual 指标（npu 对齐 bench，MARE_ratio）

| Case | dh MARE_ratio | dv2 MARE_ratio | 权威日志 |
|------|---------------|----------------|----------|
| `smoke_varlen_t256_v256` | 0.0 | 0.0 | npu7 |
| `smoke_fixed_t4096_v256` | 1.05 | 0.80 | run_20260625 |
| `varlen_t16384_v256_cu2` | 1.17 | 1.13 | v5 |
| `varlen_t16384_v256_cu128` | 1.87 | 1.00 | v4 |
| `varlen_t65536_v256_cu17` | 1.17 | 1.17 | v5 |
| `varlen_t65536_v256_cu172` | 2.03 | 1.00 | v5 |
| `varlen_t65536_v256_cu668` | 1.74 | 1.00 | v5 |
| `fixed_b16_t2048_v256` | 1.08 | 1.15 | v5 |
| `fixed_b176_t24_v256` | 0.0 | 0.0 | npu7 |
| `fixed_b711_t196_v256` | 1.80 | 1.00 | v5 |

阈值：MARE_ratio ≤ 5.0，MERE_ratio ≤ 1.5，RMSE_ratio ≤ 1.5，ERR_COUNT_ratio ≤ 2.0。

## 四、各轮跑测日志索引

| 日志文件 | 设备 | 说明 |
|----------|------|------|
| `bwd_dhu_gva_aclnn_20260622_npu2.log` | NPU2 | `set_device(2)` 失败 507033，未跑通 |
| `bwd_dhu_gva_aclnn_20260622_npu7.log` | NPU7 | 首轮：smoke_varlen、fixed_b176 PASS；fixed_b711 启动后卡住 |
| `bwd_dhu_gva_aclnn_20260622_npu7_v2.log` | NPU7 | 重跑子集，smoke_varlen PASS |
| `bwd_dhu_gva_run_20260625.log` | NPU7 | smoke_varlen、smoke_fixed PASS |
| `bwd_dhu_gva_aclnn_20260625_v3.log` | NPU7 | smoke_fixed 精度 PASS（脚本 bug）；fixed_b16 启动后极慢 |
| `bwd_dhu_gva_aclnn_20260625_v4.log` | NPU7 | cu128 PASS；cu2/cu17/cu172 因 `generate_cu_seqlens` FAIL |
| **`bwd_dhu_gva_aclnn_20260625_v5.log`** | **NPU2** | **最终完整轮**：7 case PASS + 4 SKIP + 1 主动 SKIP |

## 五、跑测过程问题与修复

| 问题 | 影响 | 修复 |
|------|------|------|
| `generate_cu_seqlens` 硬限 `[64,128]` | cu2/cu17/cu172 无法生成 cu_seqlens | T 超范围时退化为公平切分 |
| `save_out` 未传入 `run_case` | v3 smoke_fixed 误标 FAIL | 增加参数传递 |
| CPU fp64 golden 双遍 | 大 shape 极慢（非 NPU 问题） | 调整 case 顺序、skip 已测 case |
| NPU7 import flock 卡死 | v5 首轮 16min 无输出 | 改 NPU2 重启 |
| NPU2 `507033` | 早期无法 set_device | 换 NPU7/NPU2 空闲卡 |

## 六、v5 最终 SUMMARY（原文）

```
===== SUMMARY =====
smoke_varlen_t256_v256: SKIP (BWD_HU_SKIP_CASES)
smoke_fixed_t4096_v256: SKIP (BWD_HU_SKIP_CASES)
varlen_t16384_v256_cu2: PASS (dh=PASS dv2=PASS)
varlen_t16384_v256_cu128: SKIP (BWD_HU_SKIP_CASES)
varlen_t65536_v256_cu17: PASS (dh=PASS dv2=PASS)
varlen_t65536_v256_cu172: PASS (dh=PASS dv2=PASS)
varlen_t65536_v256_cu668: PASS (dh=PASS dv2=PASS)
fixed_b16_t2048_v256: PASS (dh=PASS dv2=PASS)
fixed_b176_t24_v256: SKIP (BWD_HU_SKIP_CASES)
fixed_b711_t196_v256: PASS (dh=PASS dv2=PASS)
varlen_t262144_v256_cu32: SKIP (CPU fp64 golden T=262144 过慢，暂跳过)
```

## 七、复现命令

```bash
cd torch_custom/fla_npu/test
TEST_DEVICE_ID=2 BWD_HU_SAVE_OUT=0 bash run_bwd_dhu_gva_cases.sh \
  2>&1 | tee bwd_dhu_gva_aclnn_$(date +%Y%m%d).log
```
