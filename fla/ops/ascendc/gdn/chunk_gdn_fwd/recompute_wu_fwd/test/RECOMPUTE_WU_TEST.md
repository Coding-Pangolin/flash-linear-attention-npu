# recompute_wu 双标杆测试指南

算子：`recompute_wu_fwd`（重算 w/u 前向）  
输出：`w`, `u`（及中间量，以 dual 脚本为准）

总览与共享文件清单：[`fla/ops/ascendc/gdn/GDN_DUAL_TEST_GUIDE.md`](../../../GDN_DUAL_TEST_GUIDE.md)

---

## 1. main 基础上需添加的本算子文件

在共享层（见总览 §2.1）之外，**recompute_wu 还需以下 4 个文件**：

| 文件 | 用途 |
|------|------|
| `run_recompute_wu_gpu_dump_dual.sh` | GPU dual 入口 |
| `test_recompute_wu_gpu_dump_dual.py` | 读 GPU dump，`ct.dual(npu, fp64, gpu)` |
| `run_recompute_wu_cpu_dual_casesjson.sh` | CPU dual 入口 |
| `run_recompute_wu_cpu_dual_casesjson.py` | 随机输入，`ct.dual(npu, fp64, npu_bench)` |

路径前缀：`fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test/`

---

## 2. 环境

```bash
conda activate wnc
export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
export VENDOR_SET_ENV=/path/to/fla_npu_transformer/bin/set_env.bash   # 若有
source torch_custom/fla_npu/test/setup_cann_env.sh
export TEST_DEVICE_ID=0
```

`cases.json`：`fla/ops/ascendc/gdn/cases.json`

---

## 3. GPU dual（竞品 dump 标杆）

### 3.1 前置条件

1. GPU 机 checkout `flash-linear-attention` 的 `feat/gdn-gpu-dump`，按 `GDN_DUMP_GUIDE.md` 采集 dump。
2. dump 目录 rsync 到 NPU 机，结构为 `<DUMP_ROOT>/<case_name>/004_recompute_wu.pt`（或脚本支持的 `--pt` 单文件）。

### 3.2 命令

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test

# 整目录
TEST_DEVICE_ID=0 ./run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP

# 单 case
./run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1

# 单 .pt
./run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP/phase_1_fix_1/004_recompute_wu.pt

# 跳过 viz（加速）
./run_recompute_wu_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1 --no-viz
```

### 3.3 常用参数

| 参数 | 说明 |
|------|------|
| `--dump-root DIR` | GPU dump 根目录 |
| `--case NAME` | 单个 cases.json 用例名 |
| `--cases a,b,c` | 多个用例 |
| `--phase prefix:phase_1_` | 按前缀过滤 |
| `--dump-phase fwd\|bwd\|any` | recompute 专用：读 fwd/bwd 阶段 dump |
| `--no-viz` | 不做 ct.viz |
| `-sc N` | viz 采样点数（默认 200000） |

日志：`<DUMP_ROOT>/logs/recompute_wu_gpu_dump_dual_<timestamp>.log`

---

## 4. CPU dual（随机输入，无需 dump）

覆盖 `cases.json` 中 **CPU-only 8 项** + smoke；GPU 不支持或 `chunk_size=128` 等无法 dump 的 case 走此路径。

### 4.1 命令

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_wu_fwd/test

# smoke（小 case，推荐先跑）
TEST_DEVICE_ID=0 ./run_recompute_wu_cpu_dual_casesjson.sh --smoke --no-viz

# 默认 8 项 CPU-only batch
TEST_DEVICE_ID=0 ./run_recompute_wu_cpu_dual_casesjson.sh --no-viz

# 单 case
TEST_DEVICE_ID=0 ./run_recompute_wu_cpu_dual_casesjson.sh --cases gva_fix_3 --no-viz

# 多个 case
TEST_DEVICE_ID=0 ./run_recompute_wu_cpu_dual_casesjson.sh --cases gva_var_2,phase_1_var_4 --no-viz
```

### 4.2 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TEST_DEVICE_ID` | 2 | NPU 设备号 |
| `RECOMPUTE_WU_CPU_DUAL_OUT` | `dual_benchmark_logs/recompute_wu/cpu_dual_casesjson/<ts>/` | 输出目录 |
| `ASCEND_LAUNCH_BLOCKING` | 1 | 同步 launch，便于定位错误 |

### 4.3 Python 参数

| 参数 | 说明 |
|------|------|
| `--smoke` | 内置小 case |
| `--cases LIST` | 逗号分隔 cases.json 名 |
| `--no-viz` | 跳过 ct.viz |
| `--no-save-outputs` | 不写 outputs.pt |
| `--device N` | 覆盖 TEST_DEVICE_ID |

输出：`fla/ops/ascendc/gdn/dual_benchmark_logs/recompute_wu/cpu_dual_casesjson/<timestamp>/`

---

## 5. 通过标准

- **GPU dual：** 每个输出张量 `ct.dual` L1 全 PASS（见总览 §5）。
- **CPU dual：** 同上；fp64 为 golden，npu-aligned CPU 为 bench。

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `test.py` | 算子目录 legacy 单测（非 dual） |
| `../README.md` | 算子说明 |
| `gdn_cpu_dual_casesjson.py` | 共享 CPU dual 逻辑 |
| `run_gdn_gpu_dump_dual_all.sh` | 三算子 GPU dual 串行（含本算子） |
| `run_gdn_cpu_dual_casesjson.sh --op recompute_wu` | 三算子 CPU dual 串行 |
