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
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_fwd_intra_sub_chunk \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

## 测试

- Golden：`test/test_chunk_kda_fwd_intra_sub_chunk.py`（含 GVA；可选 `score_dtype` 对标 Cube）
- 单算子：`torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py`
  - smoke：全量 golden（`score_dtype=输入 dtype`）
  - 模型级：`_run_case_model_sample`（NPU vs bf16-sim 采样，避免 T=8192 全量 CPU golden 过慢）
  - `FLA_NPU_ONLY_MODEL=1` / `FLA_NPU_ONLY_GVA=1` 可筛跑

## 实现说明

| Tiling key | 路径 | 分核 |
|------------|------|------|
| 0 | AIV scalar fallback | 外层 `B×HV×NT`，核内 NC |
| **1（默认）** | **MIX_AIC_1_2 Cube** | 外层 `B×HV×NT` + `GetCoreNumAic`，核内 NC |

**精度路径（对齐 `chunk_kda_fwd`）**

- AIV Vector：fp32 算 midpoint gate / 乘加，再 `Cast` 成输入 dtype 写入 scratch
- AIC Cube：`BlockMmad` 的 A/B 为 bf16/fp16，C 为 fp32（**不升精度**）
- AIV post：fp32 tril / β / forward-sub / store

计划与交接：`/root/.cursor/plans/intra_sub_chunk_cube_catlass_7a3f2c1b.plan.md`；分析见 `PARTITION_CUBE_ANALYSIS.md`。
