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
| **当前** | **~7.66** | vs C1 **−13.8** | 上表全 1；`L0_AB_DBUF`/`FIX_MTE2`=0 | 仍 Vec-bound |

画像（N2 后）：`aic_cube_ratio≈3.8%`；`wait V_GATE≈2.45ms`；`wait V_MASK≈1.57ms`。

---

## 2. 下一刀裁决（plan × PR190 × isub）

| 候选 | 来源 | 裁决 | 原因 |
|------|------|------|------|
| **窗 soft-lead Prefill** | plan **I5** + isub [`VEC_2WIN_PIPE.md`](../../../../../VEC_2WIN_PIPE.md) | **先做** | 已远差 0.8ms；4-slot/`slotFree` 骨架在；墙钟仍钉跨窗串行 Stage0–3 |
| I4a/b Fix∥MTE2 / L0 dbuf | plan I4 | **暂缓** | `aic_cube≈4%`，AIV-bound 时墙钟常无感 |
| 照搬 PR190 `Process` | ref_pr190 | **不优先** | PR190 窗内仍 stage 串行，**无** 2-win Prefill；已吃完其 Select/WRS/双 AIV |
| 残余 Vec BAR / Exp | plan I6 | soft-lead 后复测再开 | 边际相对 Prefill 小 |

**下一刀落地形态（I5）：**

```text
Prefill: w=0,1 的 Stage0（及必要握手）
稳态:   Store(w) ‖ Stage0(w+2)   # 同 bank：先 Store 再写
热路径: 仅 C_S* / V_* ；slotFree 只做 Process 书挡
```

红线：raw `0x2`；空头也握手；Cube/Vec 镜像；禁每窗 `WaitFree`。

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
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
```
