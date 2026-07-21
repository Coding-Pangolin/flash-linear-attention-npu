# GPU vs CPU dual: `chunk_kda_fwd_intra_sub_chunk`

对比 GPU Triton `chunk_kda_fwd_kernel_intra_sub_chunk`（`safe_gate` 对角路径）与本目录 CPU golden。

## 环境（CUDA 机）

```bash
# 已安装 GPU 版 flash-linear-attention（含 fla.ops.kda）
pip install ct
export CUDA_VISIBLE_DEVICES=0   # 按需
```

在本仓库根目录，或把 `test/` 目录拷到任意路径后执行：

```bash
python fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test/ \
  test_gpu_cpu_dual_chunk_kda_fwd_intra_sub_chunk.py --smoke

# 完整小/中 shape + ct.viz
python fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/test/ \
  test_gpu_cpu_dual_chunk_kda_fwd_intra_sub_chunk.py --viz-dir ./viz_sub_chunk

# 自定义 shape
python .../test_gpu_cpu_dual_chunk_kda_fwd_intra_sub_chunk.py \
  --B 1 --H 4 --T 256 --K 128 --BT 64 --gate lin_mild --dtype bf16
```

## Layout

| 侧 | q/k/g | beta | Aqk | Akkd |
|----|-------|------|-----|------|
| GPU Triton | `[B,T,H,K]` BSND | `[B,T,H]` | `[B,T,H,BT]` | `[B,T,H,16]` |
| CPU golden | `[B,H,T,K]` BNSD | `[B,H,T]` | `[B,H,T,BT]` | `[B,H,T,16]` |

脚本内部：构造 BNSD → 转 BSND 喂 GPU → 结果转回 BNSD 再和 CPU / `ct.viz` 对比。

## 注意

- GPU kernel 仅支持 `chunk_size ∈ {32, 64}`（本阶段 MHA，`H == HV`）。
- GPU 生产路径对 `Aqk` 用 `empty`；本脚本 **zero-init**，以便与 CPU 未写区域对齐。
- `ct.viz` 输出默认 `./viz_chunk_kda_fwd_intra_sub_chunk/`；可用 `--no-viz` 关闭。
