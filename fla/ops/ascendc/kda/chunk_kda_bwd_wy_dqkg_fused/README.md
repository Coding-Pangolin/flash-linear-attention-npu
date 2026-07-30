# ChunkKdaBwdWyDqkgFused

对标 Triton `chunk_kda_bwd_kernel_wy_dqkg_fused`。契约见 [DESIGN.md](DESIGN.md)。

## 构建

```bash
# 方式 A（推荐）：一键 wheel
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_bwd_wy_dqkg_fused \
  python -m pip wheel --no-build-isolation --no-deps . -w dist

# Ascend950 / A5（arch35 regbase vector；本机无 950 卡时仅编译门禁）
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd_wy_dqkg_fused \
  python -m pip wheel --no-build-isolation --no-deps . -w dist_950
```

备选：`bash build.sh --pkg --soc=ascend910b --ops=chunk_kda_bwd_wy_dqkg_fused`

A5 双路径说明见 [`op_kernel/arch35/README.md`](op_kernel/arch35/README.md)。  
**Ascend950 板端精度 + 性能测试步骤**：[`ASCEND950_TEST_GUIDE.md`](ASCEND950_TEST_GUIDE.md)。
## 测试

```bash
python fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/test/test_chunk_kda_bwd_wy_dqkg_fused.py
python torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_wy_dqkg_fused.py
```

## Python

```python
from fla_npu.ops import ascendc as ops
dq, dk, dv2, dg, db, dA = ops.npu_chunk_kda_bwd_wy_dqkg_fused(
    q, k, v, v_new, g, beta, a, h, dh, do, dv, scale, 64, state_v_first=False
)
```

## Stage / 流水

- Stage0 WyV → Stage1 KvAcc（Cube∥Vec kg）→ Stage2 GateWy 三明治 → Stage3 DaFinal
- 4 GM slot；raw CrossCore `0x2`
- 当前 Cube 路径为 DirectTileGemm 基线；L1 A resident / L0 dbuf / Fix∥MTE2 为后续性能刀（目标 Task Dur ≤0.8 ms）
- 性能迭代（改造方向 + 落地方案）：[PERF_ITER_PLAN.md](PERF_ITER_PLAN.md)
- 代码量 + 可行优化方向（含伪代码）：[OPT_DIRECTION.md](OPT_DIRECTION.md)
- **迭代留档（年表 + 下一刀裁决）**：[ITER_LOG.md](ITER_LOG.md)
- **G 档迭代 Plan（接 3.86 ms）**：[NEXT_ITER_PLAN_G.md](NEXT_ITER_PLAN_G.md)
- **改进策略（PR190 / Sim）**：[IMPROVEMENT_STRATEGY.md](IMPROVEMENT_STRATEGY.md)
