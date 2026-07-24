# ChunkKdaFwdIntraSubChunk

AscendC L0 算子，对标 GPU Triton `chunk_kda_fwd_kernel_intra_sub_chunk`
（`flash-linear-attention/fla/ops/kda/chunk_intra.py`）。

## 功能

对每个对角 sub-chunk（`BC=16`）：

1. Midpoint gate centering：`gm = g - g[mid]`，`exp2(±gm)`
2. `Aqk = tril((q*gq) @ (k*gk)^T) * scale`
3. `L = strict_tril((k*gq) @ (k*gk)^T) * beta`
4. fp32 forward substitution → `Akkd = (I - L)^{-1}`

## 接口

```python
from fla_npu.ops.ascendc import npu_chunk_kda_fwd_intra_sub_chunk

aqk, akkd = npu_chunk_kda_fwd_intra_sub_chunk(
    q, k, g, beta, scale, chunk_size,
    cu_seqlens=None, chunk_indices=None,
)
```

### Shape（BNSD，MHA + GVA）

| Tensor | Shape | dtype |
|--------|--------|-------|
| q, k | `[B, H, T, K]` | fp16 / bf16 |
| g | `[B, HV, T, K]` | 同 q |
| beta | `[B, HV, T]` | 同 q |
| aqk | `[B, HV, T, chunk_size]` | 同 q |
| akkd | `[B, HV, T, 16]` | float32 |

- `HV >= H` 且 `HV % H == 0`
- Head 映射与 Triton 一致：`i_h = i_hv // (HV // H)`
- MHA 即 `H == HV`

### 约束

- `chunk_size ∈ {32, 64, 128}`，`BC=16` 固定
- `K` 为 16 的倍数且 `<= 256`；`H,HV <= 128`
- dense：可 `B>1`
- varlen：`cu_seqlens` 与 `chunk_indices` **成对**出现；`B=1`；indices 扁平 `(seq_id, local_chunk_id)*`
- 公开 layout **仅 BNSD**（与 Triton BSND 不同）

### 模型目标 shape

`B=1, T=8192, H=32, K=128, chunk_size=64`（可扩展 `HV>H` 的 GVA）

## 与 GPU 差异

| 项 | GPU Triton | 本算子 |
|----|------------|--------|
| layout | BSND `[B,T,H/HV,K]` | BNSD `[B,H/HV,T,K]` |
| chunk_size | 32 / 64 | 32 / 64 / **128** |
| GVA | 支持 | **支持**（`HV%H==0`） |

## 构建

```sh
# A2/A3（精度 + 可跑）
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_fwd_intra_sub_chunk \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl

# A5 / Ascend950（首版：编译门禁；与 A2 同一套源码，CATLASS_ARCH 切换）
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_fwd_intra_sub_chunk \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
```

## 测试

- Golden：`test/test_chunk_kda_fwd_intra_sub_chunk.py`（含 GVA；可选 `score_dtype` 对标 Cube）
- 单算子：`torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py`
  - smoke：全量 golden（`score_dtype=输入 dtype`）
  - 模型级：`_run_case_model_sample`（NPU vs bf16-sim 采样，避免 T=8192 全量 CPU golden 过慢）
  - `FLA_NPU_ONLY_MODEL=1` / `FLA_NPU_ONLY_GVA=1` 可筛跑

### 性能（Phase E，未结案）

- **硬目标**：模型 case `B=1,T=8192,H=HV=32,K=128,BT=64,bf16` 的 msprof **Task Duration 中位 ≤ 1.5 ms**
- 当前板端（P1+C1 path A）约 **~2.18 ms**；Vector 侧优化待做
- 板端：`torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py` + `msprof op ... --replay-mode=application`
- 流水仿真（T=1024，不替代 1.5 ms 验收）：
  ```sh
  export LD_LIBRARY_PATH=/data/tracy/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
  msprof op simulator --soc-version=Ascend910B3 python \
    torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py
  ```
- 迭代记录：`PERF_ITER_LOG.md`；Cube 理论最优路径：`CUBE_OPTIMAL_PIPELINE.md`（**路径 A**：`L1A_DBUF=1` + `FIX_MTE2_DBUF=1`，`WIN_L1_RESIDENT=0`）

## 实现说明

| Tiling key | 路径 | 分核 |
|------------|------|------|
| 0 | AIV scalar fallback | 外层 `B×HV×NT`，核内 NC |
| **1（默认）** | **MIX_AIC_1_2 Cube** | 外层 `B×NT` + dual-AIV-by-head（Vec2Win）+ AIC |

**精度路径**

- AIV Vector：fp32 算 midpoint gate / 乘加，再 `Cast` 成输入 dtype 写入 scratch
- AIC Cube：Tile GEMM（A/B bf16/fp16，C fp32）；**Kg→L1B 驻留**；**L1A 双槽**（Qg/W，`MTE2(W)‖MMAD1`，Wait W 再 Fix Aqk）；**C1** Akk Fix ‖ 下 tile MTE2
- AIV post：fp32 tril / β / forward-sub / store（**无 MCH**）

计划：`/root/.cursor/plans/intra_subchunk_impl_c485e88b.plan.md`。
