# ChunkKdaBwdWyDqkgFused — 迭代留档

> Shape：`B1 H=HV=32 T8192 K128 V128 BT64 bf16`，`state_v_first=false`  
> 指标：裸 `msprof op` → MIX_AIC **Task Duration med**  
> 门禁：suite 全绿 + Δ ≤ **−0.05 ms** 才 default on  
> 硬目标（stretch）：≤ **0.8 ms**

约定：**每刀有效优化 → 更新本表 + git commit**（不含 `dist/` / opp 产物）。

---

## 1. 年表

| 阶段 | med ms | Δ vs 前 | 宏 default | 备注 |
|------|--------|---------|------------|------|
| C0 早期基线 | ~29–32 | — | — | 单 head / 调度未成组 |
| 双 AIV 半行修好 | ~22.0 | — | — | contig split，无 Set Barrier |
| C1（I1+I2） | **21.48** | −0.52 vs ~22 | `L1_A_RESIDENT` `STAGE1_L0C_ACCUM` | Cube 仍等 Vec |
| V1 Gate reuse | 21.40 | −0.07 | `GATE_REUSE_KG_WS` | park k/g |
| V2 Epilog fold | **16.60** | **−4.8** | `EPILOG_VEC_FOLD` | PR190 WholeReduceSum；去 Get/SetValue |
| V3 Dual Mask + early | 16.41 | −0.19 | `DUAL_AIV_MASK` `EARLY_MASK` | |
| V4 Stage2 preload A | （并入） | — | `STAGE2_PRELOAD_A` | Gate 空泡仍大时开 |
| V5 Stage0 dA L0C | **8.29** | 大段来自 V2 修齐后 | `STAGE0_DA_L0C_ACCUM` | |
| N1 Gate early Set | 7.82 | −0.28 vs 8.10 | `GATE_EARLY_SET` | dwNeg 后立刻 Set；merge∥Stage2 |
| N3 Dual Store | （并入 N1） | — | `DUAL_AIV_STORE` | |
| N2 Mask soft-lead | **7.66** | −0.16 vs 7.82 | `MASK_SOFT_LEAD` | 末 BK：dA→Mask/Set→state |
| **I5 窗 Prefill** | **7.27** | **−0.39** | `WIN_SOFT_LEAD=1` `PREFILL=2` | suite 绿；`aic_cube≈4.0%` |
| **当前** | **7.27** | vs C1 **−14.2** | 上表；`L0_AB_DBUF`/`FIX_MTE2`=0 | 仍 Vec-bound |

I5 画像：Task Dur **7271 µs**；`aic_cube_ratio≈4.0%`；scalar≈40%；MTE2≈27%。

---

## 2. 下一刀裁决

| 候选 | 裁决 | 原因 |
|------|------|------|
| **I5b** Cube Post(w+1) 先于 WaitFree(w)+S0(w+2) | **先做** | 现序在 Post(w) 后立刻 WaitFree，挡住 Post(w+1) |
| I6 Vec BAR / Exp | I5b 后 | 仍 AIV-bound |
| I4a/b Fix∥MTE2 / L0 dbuf | 试刀可开 | cube 仍低，预期无感但成本低 |

I5 已落地（对标 isub，非 PR190 Process）：

```text
Prefill: Stage0(w=0,1) 实灌
稳态 Vec: Post(w) → Stage0Vec(w+2)
稳态 Cube: Post(w) → WaitFree(bank w) → Stage0(w+2)
SetVS0Joined: CrossCoreBarrier（防 Prefill 0x2 skew）
```

---

## 3. 已开宏一览（kernel）

```text
USE_L1_A_RESIDENT=1
USE_STAGE1_L0C_ACCUM=1
USE_GATE_REUSE_KG_WS=1
USE_EPILOG_VEC_FOLD=1
USE_DUAL_AIV_MASK=1
USE_EARLY_MASK_PER_HEAD=1
USE_STAGE2_PRELOAD_A=1
USE_STAGE0_DA_L0C_ACCUM=1
USE_GATE_EARLY_SET=1
USE_DUAL_AIV_STORE=1
USE_MASK_SOFT_LEAD=1
USE_WIN_SOFT_LEAD=1
KDA_BWD_PREFILL_WINDOWS=2
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
```
