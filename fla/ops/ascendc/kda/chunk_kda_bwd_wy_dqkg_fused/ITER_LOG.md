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
| V1–V5 / N1–N3 | **7.66** | vs C1 −13.8 | 见下行宏 | Epilog fold 为大头 |
| **I5 窗 Prefill** | **7.26** | **−0.40** | `WIN_SOFT_LEAD=1` `PREFILL=2` | suite 绿；复测 7259 µs |
| I5b Post≻WaitFree | skip | — | — | 推迟 Cube S0(w+2) 会堵 Vec（AIV-bound） |
| I6 Kg∥Gate 交织 | reject | **+0.06** | `KG_GATE_INTERLEAVE=0` | 7333 µs |
| I4a FIX∥MTE2 | parked | — | `FIX_MTE2=0` | model FIXP/L0C ECC；已修 evt=14 + outstanding 状态机 |
| I4b L0 dbuf | parked | — | `L0_AB_DBUF=0` | 与 I4a 同开曾 ECC |
| Epilog kPark 复用 | reject | **+6** | — | 13348 µs；已回滚 |
| VS0 每窗一次 | reject | **+0.12** | `VS0_ONCE=0` | 7375 µs |
| **Epilog state panel** | **done** | **−1.29** | fold 内 | **5.97 ms**；[K,V] 整面板 CopyStrided |
| EXP_GN_PARK | reject | **+0.06** | `=0` | 6029 µs；Exp 挪到 Kg 无净收益 |
| **当前** | **5.97** | vs C1 **−15.5** | 下表 | 仍 Vec-bound；距 0.8 尚远 |

画像（state panel 后）：Task Dur **5971 µs**。

---

## 2. 下一刀裁决

| 候选 | 裁决 | 原因 |
|------|------|------|
| Gate/Epilog MTE2∥V ping-pong | **next** | PR190 InitVectorEvents；攻 MTE2 |
| 更深跨窗双发 Post | 高风险 | 需重做 stage 信用 |
| I4 FIX∥MTE2 | parked | model ECC |
| PR190 Process 照搬 | 不做 | 无 Prefill |

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
USE_KG_GATE_INTERLEAVE=0
USE_VS0_ONCE_PER_WINDOW=0
USE_L0_AB_DBUF=0
USE_FIX_MTE2_OVERLAP=0
```
