# Intra Sub-Chunk GPU Dump 双标杆测试指南

> 分支：`20260725_105513_intra-sub-chunk-gpu-dump-dual`（基于 0723 Vec2Win 最优实现）  
> 参考：`ref_pr190/.../GDN_DUAL_TEST_GUIDE.md`  
> GPU 采集：FLA 仓 `20260724_222750_intra-sub-chunk-gpu-dump`

---

## 0. 布局（必读）

| 侧 | Layout | q/k | g / aqk / akkd | beta |
|----|--------|-----|----------------|------|
| **GPU dump** | **BTHD** | `[B,T,H,K]` | `[B,T,HV,*]` | `[B,T,HV]` |
| **CPU dump** | **BTHD**（与 GPU 一致） | 同上 | 同上 | 同上 |
| **NPU aclnn** | **BNSD** | `[B,H,T,K]` | `[B,HV,T,*]` | `[B,HV,T]` |

结论：

1. **GPU layout 与 CPU dump layout 一致**（都是 BTHD；CPU 在 dump 脚本里已从内部 BNSD 转成 BTHD 落盘）。
2. **喂给 NPU 必须转置**：`transpose(1, 2)`（BTHD → BNSD）。  
   一键脚本的 `isub_gpu_dump_loader.py` **已自动转置**，业务侧不必手转。

---

## 1. 准备

### 1.1 GPU 机采集

见 FLA 仓 `INTRA_SUB_CHUNK_DUMP_GUIDE.md`：

```bash
./run_intra_sub_chunk_dump_cases.sh --phase smoke --dump-dir /data/isub_dump/smoke
# 或全量
./run_intra_sub_chunk_dump_cases.sh --dump-dir /data/isub_dump/gdn
```

每个 case 目录应有：

```text
case_meta.json
manifest.json
001_chunk_kda_fwd_intra_sub_chunk.pt   # inputs + aqk/akkd + aqk_cpu/akkd_cpu
002_chunk_kda_fwd_intra_sub_chunk_cpu.pt  # 可选
```

拷到 NPU 机，例如 `/data/isub_gpu_dump/gdn`。

### 1.2 NPU 环境

```bash
conda activate wnc
export CANN_SET_ENV=/data/wnc/cann/ascend-toolkit/set_env.sh   # 按本机改
# 可选 vendor / OPP
export TEST_DEVICE_ID=0
pip install ct
# 算子已按本仓 0723 最优实现编译安装
```

---

## 2. 推荐：同种子 CPU 随机（无需拷 dump）

两边用**同一份** `intra_sub_chunk_cases.json`、同一 `--seed` / `--phase` / `--names`：

| 侧 | 命令 |
|----|------|
| GPU | `./run_intra_sub_chunk_seed_dual.sh --phase smoke --seed 0` |
| NPU | `./run_isub_seed_dual.sh --phase smoke --seed 0` |

规则：

1. 在 **CPU** 上按 **BTHD** 采样（与 GPU dump 脚本默认一致）
2. NPU：`transpose(1,2)` → **BNSD** 再调算子
3. `seed_i = base_seed + index * 9973`（`index` = **过滤后** case 列表下标）

**CPU 标杆开关（两边相同语义）：**

| 选项 | 含义 |
|------|------|
| （默认）`|--run-cpu` | 跑 CPU golden（与加速器对比） |
| `--no-cpu` | 不跑 CPU，只跑 GPU/NPU |
| `--cpu-only` | 只跑 CPU（GPU 侧可不需 CUDA） |
| `--cpu-dtype fp32\|fp64` | CPU 计算精度（默认 fp32） |
| `--save-cpu` | 落盘 `cpu_golden.pt` |

```bash
# NPU
TEST_DEVICE_ID=0 ./run_isub_seed_dual.sh --phase smoke --seed 0
./run_isub_seed_dual.sh --phase smoke --seed 0 --cpu-only --save-cpu
./run_isub_seed_dual.sh --phase smoke --seed 0 --no-cpu

# GPU（同 seed / phase）
./run_intra_sub_chunk_seed_dual.sh --phase smoke --seed 0
./run_intra_sub_chunk_seed_dual.sh --phase smoke --seed 0 --cpu-only --save-cpu
```

---

## 3. 备选：从 GPU dump 读输入（旧流程）

```bash
chmod +x run_isub_gpu_dump_dual.sh
TEST_DEVICE_ID=0 ./run_isub_gpu_dump_dual.sh /data/isub_gpu_dump/gdn
./run_isub_gpu_dump_dual.sh /data/isub_gpu_dump/gdn --names smoke_mha_fix
./run_isub_gpu_dump_dual.sh --pt /data/.../001_chunk_kda_fwd_intra_sub_chunk.pt
```

### 对比内容（seed 模式）

1. CPU RNG 生成 BTHD → 转 BNSD  
2. 本地 CPU golden + NPU  
3. 有 `--gpu-dump-root` 时再 `ct.dual(npu, cpu, gpu)`

---

## 4. 输出（seed 模式）

```text
./isub_seed_dual_out/
  intra_sub_chunk_seed_dual_report.json
  logs/seed_dual_<ts>.log
  viz/<case>/aqk*_Standard.png
```

---

## 5. 文件清单

```text
fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test/
  intra_sub_chunk_cases.json          # 与 GPU 仓同步
  isub_seed_case_utils.py             # CPU RNG + BTHD 构造
  test_intra_sub_chunk_seed_dual.py   # 推荐入口
  run_intra_sub_chunk_seed_dual.sh
  isub_gpu_dump_loader.py             # dump 模式仍可用
  test_intra_sub_chunk_gpu_dump_dual.py
  INTRA_SUB_CHUNK_DUAL_TEST_GUIDE.md
run_isub_seed_dual.sh                 # 仓根一键（seed）
run_isub_gpu_dump_dual.sh             # 仓根一键（dump）
```
