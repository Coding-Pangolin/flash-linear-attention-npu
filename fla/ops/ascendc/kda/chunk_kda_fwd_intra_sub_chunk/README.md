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

- Golden：`test/test_chunk_kda_fwd_intra_sub_chunk.py`（含 GVA）
- 单算子：`torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py`（安装后）

## 实现说明

当前 kernel 为 AIV 路径（BC×K 小矩阵累加 + 16×16 forward-sub）；task 维按 `B*HV*NT*NC` 调度。后续可按 `chunk_kda_fwd` 的 Catlass cube 模式升级 GEMM 热路径。
