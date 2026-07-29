# ChunkKdaBwdWyDqkgFused — 性能迭代 Plan

> 对标 Triton `chunk_kda_bwd_kernel_wy_dqkg_fused`。  
> 契约：[DESIGN.md](DESIGN.md)（**P0 需与代码对齐**）  
> 代码：`op_kernel/chunk_kda_bwd_wy_dqkg_fused_{common,cube,vector}.h`  
> 约束：不融合拆算子 · 910B 首发 · **单变量宏** · 精度 suite → msprof Δ ≤ **−0.05 ms** 才 default on  
> 硬目标（stretch）：model Task Dur med ≤ **0.8 ms**（裸 msprof，筛 MIX_AIC）  
> 关联经验：isub `VEC_2WIN_PIPE.md` / `SCORE_TILE_DBUF_PLAN.md` / `STAGE_OPT_ITER_PLAN.md`

---

## 0. 一句话

调度骨架（2-head 成组、`Kg∥Stage1`、4-slot、`V_S0`）已合理；墙钟大头在 **Cube 每 V-tile 一次 Fixpipe + Vec 再 Add 归约**。  
本轮主路径：**画像钉基线 → 打开已有 L1 A resident → Stage1/Stage0 L0C 累加砍 Fix → 再开 Fix∥MTE2 / 窗 soft-lead**。  
禁止回退 DESIGN §5.2 三反模式；禁止无画像并行乱开宏。

---

## 1. 现状快照（代码，2026-07-28）

### 1.1 已落地（正确性向）

| 项 | 状态 |
|----|------|
| Stage0 WyV / Stage1 KvAcc / Stage2 GateWy / Stage3 DaFinal | 成组调度，Cube/Vec 镜像 |
| `KgVec` 与 Cube Stage1 **无互 wait** | 已实现 |
| `FLAG_V_S0`：Stage0Vec 完成后再进 Stage1 | Cube `Wait(vS0_)` / Vec `SetVS0Joined` |
| 4 GM slot，`windowIdx&1` bank，`windowIdx≥2` 等 `slotFree` | ping-pong 骨架已有 |
| Dual AIV 半行 + `dbMergeWs` / `dgkMergeWs` | `kSingleWriterAiv=false` |
| Gate 归约优先 `WholeReduceSum` | 已减 scalar 尾 |
| FFTS flag id | **`0..7,11..13`**（跳过 Catlass 保留 **8/9/10**） |
| `SetV*Joined` | **仅** `PipeBarrier(MTE3)+Set(0x2)`（无 AIV↔AIV `0x1`） |

### 1.2 性能基线形态（待 C0 实测）

| 项 | 现状 |
|----|------|
| Cube GEMM | `DirectTileGemm`；`tileMmad(..., /*init*/ true, 0)` → **每调用清 L0C + 一次 Fix** |
| Stage0 | 每 `iv`：`dv@vᵀ` Fix→`dASlot[iv]`；每 `iv`：`A@dv` Fix→`dvbWs` panel；Vec Σ `dASlot` |
| Stage1 | 每 `iv`×{dq,dk,dw} 各一次 Fix→`dq/dk/dwSlot[iv]`；Gate 内 `for iv: Add` |
| Stage2/3 | 每 BK / 每 head 固定 2 次 GEMM |
| 宏 | `USE_L1_A_RESIDENT` / `USE_L0_AB_DBUF` / `USE_FIX_MTE2_OVERLAP` **全默认 0** |
| DESIGN.md | Flag 表仍为旧 ID，缺 `V_S0` → **与代码不一致** |

### 1.3 模型 case 工作量（推断）

```text
B1 H=HV=32 T8192 K128 V128 BT64
→ nBk=2, nBv=2
每 head:
  Stage0: 2×(dv@vᵀ) + 2×(A@dv)     + Vec sum dA
  Stage1: 每 BK  2×3 = 6 次 Fix     + Gate 6 次 MTE2 Add
  Stage2: 每 BK  2 次 GEMM
  Stage3: 2 次 GEMM
```

相对 isub Stage-1（同 shape 优化后 ~2.27 ms），本算子 GEMM/Fix 次数明显更多；**0.8 ms 为 stretch**，中间门禁用相对 Δ。

### 1.4 已知实现细节（改码时勿踩）

- `DirectTileGemm` 内部 HardEvent **`evt=9`**：与 Catlass 保留 8/9/10 冲突风险 → I2 一并迁到安全 id。  
- partial BT：`A@dv` 曾因 Fix 进 strided panel 全 0 → 现用 contiguous `dvbWs[iv]`；`validRows<BT` 仍走 Vec `ComputeDvbPartial`。  
- `dqGatedWs`：gated dq 暂存，避免下一 BK 覆盖 `dqSlot`。  
- `betaWs`：Stage0Vec 已 park beta（DESIGN/注释若仍写 unused 需改）。

---

## 2. 瓶颈判断与改造方向分析

### 2.1 为什么「再填 AIV 空窗」可能不够

isub P1b 教训：墙钟钉在 **Fixpipe + MTE2 往返** 时，往 MCH/Score 空窗塞 Store/Prologue，Δ 常 **&lt;0.05 ms**。  
本算子 Stage1 对每个 V-tile **独立 Fix 到 GM**，再由 Gate **Load+Add**——税在 **Fix 次数 × 中间面带宽**，不在「有没有 Kg∥Stage1」（该重叠已做对）。

### 2.2 方向优先级

| 优先级 | 方向 | 砍什么 | 期望 |
|--------|------|--------|------|
| **P0** | 契约对齐 + C0 画像 | 文档/基线 | 可测 |
| **I1** | L1 `A` resident | 重复 Nd2Nz(A) | 小～中 |
| **I2** | Stage1 L0C 跨 `iv` 累加 | Fix×nBv、Gate Add×nBv | **大** |
| **I3** | Stage0 `dA` L0C 累加 | Fix×nBv、Vec Σ dA | 中 |
| **I4** | Fix∥下一 tile MTE2 / L0 dbuf | 管道气泡 | 中（需仿真） |
| **I5** | Gate 空泡 / 窗 soft-lead | Cube 等 V_GATE、跨窗空等 | 中～大（改动大） |
| **I6** | Vec 减 BAR / 合并 Wait | AIV instr 税 | 中（AIV-bound 时） |

### 2.3 与 isub 有效手段的对应

| isub 有效刀 | 本算子迁移 |
|-------------|------------|
| 换错路径（Cube MCH→Vector） | 此处无等价「换算法」；GEMM 必须留 Cube |
| Vec 2-win soft-lead | → **I5**（有 4-slot，缺 Prefill 深度） |
| Score L1A DBUF：`MTE2∥MMAD` | → **I4** + I2 后的 tile 流水；重叠面勿叠错 Fix |
| 单变量 + Δ≤−0.05 | 全文沿用 |

---

## 3. 门禁与环境

| 项 | 约定 |
|----|------|
| 精度 | `torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_wy_dqkg_fused.py`（dense / varlen / `state_v_first` / partial） |
| Prof 脚本 | `torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_model.py` |
| Shape | `B1 H32 HV32 T8192 K128 V128 BT64 bf16`，`state_v_first=false` |
| 指标 | 裸 `msprof --aic-metrics=PipeUtilization` → `op_summary` 中本算子 **MIX_AIC Task Duration 中位** |
| 规则 | 精度绿 + **Δ ≤ −0.05 ms** vs 当时 default → `default on`；否则宏回 0、代码保留 |
| 环境 | `source …/set_env.sh` + `conda activate fzy_atk`；`ASCEND_RT_VISIBLE_DEVICES=<phys>` + `ASCEND_DEVICE_ID=0` |
| 调用 | `msprof -- python path/to/script.py`（勿 `python -c`，防假 `Operation not permitted`） |

```bash
cd /workspace/fzy/code/kda/0723/flash-linear-attention-npu
source /data/wnc/cann/ascend-toolkit/set_env.sh
conda activate fzy_atk
# source fla_npu set_env.bash if needed

export ASCEND_RT_VISIBLE_DEVICES=<idle>
export ASCEND_DEVICE_ID=0

OUT=fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/results/prof_wy_c0
mkdir -p "$OUT"
msprof --aic-metrics=PipeUtilization --output="$OUT" -- \
  python torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_model.py \
  2>&1 | tee "$OUT/msprof_run.log"
```

读中位：

```bash
python3 - <<'PY'
import csv, glob, statistics
paths = sorted(glob.glob(
  "fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/results/prof_wy_c0/"
  "PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
assert paths, "no op_summary"
rows = list(csv.DictReader(open(paths[-1])))
td = sorted(float(r["Task Duration(us)"]) for r in rows
            if "WyDqkg" in r["Op Name"] or "ChunkKdaBwdWy" in r["Op Name"]
            if r["Task Type"] == "MIX_AIC")
print(paths[-1])
print(f"n={len(td)} med={statistics.median(td)/1000:.3f} ms "
      f"min={td[0]/1000:.3f} max={td[-1]/1000:.3f}")
PY
```

仿真（调度 / 管道）：短 T（如 1024）`msprof op simulator`，看 Kg vs Wait S1、Fix 次数、BAR 占比。

---

## 4. P0 — 契约同步 + C0 基线

### 4.1 DESIGN.md 对齐代码

当前权威 flag（`common.h`）：

| Flag | 值 | 方向 | 语义 |
|------|-----|------|------|
| C_S0 | 0 | Cube→Vec | Stage0 dA/dvb ready |
| C_S1 | 1 | Cube→Vec | Stage1 dq/dk/dw ready |
| V_GATE | 2 | Vec→Cube | gate dwNeg/kg ready |
| C_S2 | 3 | Cube→Vec | Stage2 dADelta/dkgb ready |
| V_MASK | 4 | Vec→Cube | masked dA ready |
| C_S3 | 5 | Cube→Vec | dA@A@A ready |
| **V_S0** | **6** | **Vec→Cube** | **Stage0Vec done（进 Stage1 前）** |
| SLOT_FREE0..3 | 7,11,12,13 | Vec→Cube | slot 可复用 |

伪代码补全：

```text
for h: Cube RunStage0; Set C_S0
for h: Vec Wait C_S0; Stage0Vec; Set V_S0
for h: Cube Wait V_S0
for iK:
  ...
```

`SetV*Joined` = `PipeBarrier(MTE3) + Set(0x2)`（文档勿再写必带 `Barrier(0x1)`）。

### 4.2 C0 产物

- `results/prof_wy_c0/SUMMARY.md`：med Task Dur、host wall、`aic_fixpipe` / `aic_mte2` / `aic_mac`  
- PEM：同 window 两 head `C_S0/C_S1` 成对；Kg 活动早于 Wait S1  

**Checklist**

- [ ] DESIGN flag / V_S0 / Joined 语义已改  
- [ ] C0 msprof med 已记录  
- [ ] 本文年表 §10 填入 C0  

---

## 5. I1 — `USE_L1_A_RESIDENT`（已实现，先关门禁）

### 5.1 问题

Stage0 每个 V-tile `A@dv` 都从 GM Nd2Nz `A`；`A` 在 tile 间不变。

### 5.2 现状

`cube.h` `#if USE_L1_A_RESIDENT`：首 tile 加载 A，后续 `skipLoadA=true`。  
`DirectTileGemm(..., skipLoadA)` 已支持。

### 5.3 落地

1. 仅将该宏改为 `1`（其余保持 0）。  
2. 精度 suite。  
3. msprof vs C0；Δ≤−0.05 → default on。  

### 5.4 风险

- 仅 full BT 走 Cube `A@dv`；partial 仍 Vec——保持。  
- L1 容量：`BT×BT` bf16 + `BT×BV` 同驻，确认 Resource 布局不踩。  

### 5.5 延伸 I1b（可选，另刀）

Stage2 `A@dwNeg`、Stage3 两拍也读 `A`。  
新宏 `USE_L1_A_RESIDENT_STAGE23`：在 Wait V_GATE / V_MASK 后复用或 EnsureResident。  
**勿与 I1 同 PR**，避免归因混乱。

**Checklist**

- [ ] 宏试验 =1  
- [ ] 精度绿  
- [ ] Δ 记录；决定 default  

---

## 6. I2 — Stage1 V-tile L0C 累加（主结构刀）

### 6.1 问题

```text
for iv:
  GEMM → Fix → dqSlot[iv]   # dk/dw 同理
Gate:
  for iv: Load Slot[iv]; Add → Acc
```

`DirectTileGemm` 每次 `initC=true`，**无法**跨 `iv` 累加。  
V=128 → 每 BK **6×Fix + Gate 6×MTE2 Add**。

### 6.2 目标

```text
for op in {dq, dk, dw}:
  for iv:
    MMAD → L0C   # iv==0 init；else accumulate
  一次 Fix → 单平面 AccWs[BT,BK]
Gate: 一次 Load AccWs（删除 iv Add 循环）
```

对齐 Triton：`b_dq += dot(...)` 寄存器累加。

### 6.3 实现要点

| 项 | 选择 |
|----|------|
| 宏 | `USE_STAGE1_L0C_ACCUM`（新），默认 **0** |
| API | 扩展 `DirectTileGemm` 或 `DirectTileGemmAccum(initC, doFix, skipLoadA)` |
| WS | 平面 0 作最终累加面，或显式 `dqAccWs`（注意与 `dqGatedWs` 生命周期） |
| Gate | `#if USE_STAGE1_L0C_ACCUM` 单次 copy；`#else` 旧路径 |
| Event | **迁出 evt=9** → 安全 id（如 14/15 若未占用；或与 flag 错开的 HardEvent 空间） |
| 单变量 | 本刀只改 Stage1 累加 + Gate 读；不开 I1b/I4 |

### 6.4 红线

- `state_v_first` true/false 两套 layout 均测。  
- partial `validRows`、尾 BV 与 golden 一致。  
- 三个输出 `{dq,dk,dw}` 之间必须完整 Fix，禁止 L0C 串污染。  
- 保持 `Kg∥Stage1`：累加不引入新的 Wait(C_S1) before Kg。  

### 6.5 验收

精度全绿；msprof vs 当时 default；记 Fix 相关 pipe 指标是否下降。

**Checklist**

- [ ] Accum API + 安全 evt  
- [ ] Stage1 三路累加路径  
- [ ] Gate 单平面读  
- [ ] 精度 + msprof + 年表  

---

## 7. I3 — Stage0 `dA` L0C 累加

### 7.1 问题

```text
for iv: dv@vᵀ → Fix → dASlot[iv]
Stage0Vec: dAWs = Σ dASlot[iv]
```

### 7.2 目标

```text
for iv: MMAD accum → 一次 Fix → dAWs
Vec: 跳过 Σ（仍做 dv2/db）
```

### 7.3 宏与范围

- `USE_STAGE0_DA_L0C_ACCUM`，默认 0。  
- **先 full BT**；partial 保持 per-tile 或 Vec 路径。  
- 与 I1 resident（`A@dv`）正交，可叠加但分 PR。  

### 7.4 风险

历史：partial + 错误 panel 写入曾导致全 0。累加只动 `dA` 平面，**不动**已修好的 `dvbWs` contiguous panel 语义。

**Checklist**

- [ ] full BT 累加  
- [ ] partial 回归不破  
- [ ] 精度 + msprof  

---

## 8. I4 — `USE_FIX_MTE2_OVERLAP` / `USE_L0_AB_DBUF`

### 8.1 现状

`DirectTileGemmPipeState` + 两宏骨架已在 `common.h`；默认关。

### 8.2 正确重叠面（isub 教训）

| 宏 | 应对齐 | 禁止 |
|----|--------|------|
| `USE_FIX_MTE2_OVERLAP` | **下一 tile MTE2 ∥ 本 tile Fix**（primed 状态机） | 与仍占用的 L1 冲突；假叠在错误 Wait 上 |
| `USE_L0_AB_DBUF` | L0A/B ping-pong 藏 MTE1 | AIV-bound 时强开（墙钟常无感） |

### 8.3 落地顺序

1. **I2 之后**再开（tile/Fix 次数已变）。  
2. 仿真确认 operand∩MMAD / Fix 空窗。  
3. **分两次**：先 `FIX_MTE2`，再 `L0_DBUF`；禁止同刀双开。  
4. 与 I2 共用「安全 evt」改造。  

**Checklist**

- [ ] I4a Fix∥MTE2  
- [ ] I4b L0 dbuf  
- [ ] 各记 Δ  

---

## 9. I5 — Gate 空泡与窗 soft-lead

### 9.1 Gate 三明治空泡

```text
Cube: Stage1(all h) ──空？── Wait V_GATE → Stage2
Vec:  Kg(all h) → Wait C_S1 → Gate → Set V_GATE
```

Kg∥Stage1 已正确。Stage1 结束后 Cube 可能空等 Gate。

**轻量刀（优先）**

- `USE_STAGE2_PRELOAD_A`：Wait V_GATE **前**只搬 `A` 入 L1（不算）。  
- PEM 量 `C_S1→V_GATE` 空隙；若 credit 下 Gate(h0) 已与 Stage1(h1) 重叠，则不必改循环。  

### 9.2 窗 soft-lead（大改，对标 isub 2-win）

现状：4 slot + `slotFree`，但 **无 Prefill**；窗内 Stage0–3 串完才 `++windowIdx`。

目标稳态：

```text
Prefill: w=0,1 的 Stage0（及必要 Stage1）
稳态:   Store(w) ‖ Stage0(w+2)   # 同 bank：先 Store 再写 score/ws
热路径: 仅 C_S* / V_* ；SetFree 只做 Process 书挡
```

**红线（抄 isub）**

- raw `0x2`；空头也参与握手  
- 禁止每窗 `WaitFree`  
- Cube/Vec 必须镜像；单独文档死锁清单  

**前置**：I1–I4 后仍远差 0.8 ms 再开；**独立 PR / 独立 plan 小节**。

---

## 10. I6 — Vector 减税

在 I2/I3 删除归约工作量之后，若仿真仍 **vec duration ≫ cube**：

1. 合并 Gate/Epilog 同依赖链 `PipeBarrier<PIPE_V>`。  
2. Store 尾 Wait 与下一窗 MTE2 重叠（UB 不冲突）。  
3. 禁止为减 BAR 而锁步降 dual-AIV / 改回单写（吞吐回退）。  

精度回归用 `DumpTensor` / `printf` 正向查。

---

## 11. 明确不做

| 项 | 原因 |
|----|------|
| 单 head 跑完 Stage0–3 再下一 head | §5.2-1，杀死成组重叠 |
| Kg 放到 Wait(C_S1) 之后 | §5.2-2 |
| 双 AIV 同算同写共享 GM | §5.2-3 |
| 无 C0 画像并行开 I1+I2+I4 | 归因失败 |
| 靠收窄 range / 砍大 T / 放宽 atol 过门 | DESIGN §4 |
| 一上来 soft-lead 重写 Process | 应先吃 Fix/归约税 |

---

## 12. PR 切分建议

```text
PR0  DESIGN 同步 + C0 SUMMARY（无行为变化）
PR1  I1 USE_L1_A_RESIDENT 门禁
PR2  DirectTileGemm：安全 evt + Accum API（默认行为不变）
PR3  I2 USE_STAGE1_L0C_ACCUM + Gate 单平面
PR4  I3 USE_STAGE0_DA_L0C_ACCUM
PR5  I4a FIX_MTE2_OVERLAP
PR6  I4b L0_AB_DBUF
PR7  （可选）I5 preload / soft-lead
```

构建：`bash build.sh --pkg --soc=ascend910b --ops=chunk_kda_bwd_wy_dqkg_fused`

---

## 13. 每刀 Checklist

1. 只开一个新宏（或 P0 纯文档）。  
2. 不破坏 `Kg∥Stage1`、2-head 成组、`V_S0`。  
3. HardEvent / CrossCore flag 不撞 8/9/10，不复用冲突 id。  
4. 精度 suite 全绿。  
5. msprof med → 填 §14 年表；Δ&gt;−0.05 → default 0。  
6. 更新本文件年表 +（若改契约）DESIGN.md。  

---

## 14. 年表

> 细粒度 Vec 刀（V1–V5 / N1–N3）见 [ITER_LOG.md](ITER_LOG.md)。本表只记 Cube/结构大阶段。

| 阶段 | 状态 | med ms | Δ | 默认 | 备注 |
|------|------|--------|---|------|------|
| Plan 本文 | **done** | — | — | — | 2026-07-28 |
| P0 DESIGN+C0 | **done** | ~22 | — | — | 双 AIV 半行 |
| I1 L1 A resident | **done** | 并入 C1 | | 1 | |
| I2 Stage1 L0C accum | **done** | **21.48** | −0.52 | 1 | |
| I3 Stage0 dA accum | **done** | 并入 ~8.3 | | 1 | + Epilog fold 等 |
| Vec 刀（V*/N*） | **done** | **~7.66** | vs C1 −13.8 | 见 ITER_LOG | 仍 Vec-bound |
| I4a Fix∥MTE2 | **parked** | | | 0 | AIV-bound 暂缓 |
| I4b L0 dbuf | **parked** | | | 0 | 同上 |
| I5 soft-lead Prefill | **next** | | | | 对标 isub 2-win；冲 0.8 |
| I6 Vec BAR | deferred | | | | Prefill 后复测 |

---

## 15. 与现有文档

| 文档 | 关系 |
|------|------|
| [DESIGN.md](DESIGN.md) | 冻结契约；P0 对齐 flag/`V_S0` |
| [README.md](README.md) | 构建/测试入口；性能状态链到本文 |
| isub `VEC_2WIN_PIPE.md` | soft-lead / SetFree 书挡 / 反模式 |
| isub `SCORE_TILE_DBUF_PLAN.md` | MTE2∥MMAD vs 错叠 Fix |
| isub `STAGE_OPT_ITER_PLAN.md` | 单变量门禁流程模板 |

---

## 16. 总结

**改造主线（更新）**：I1–I3 + Vec 刀已把墙钟从 ~22 ms → **~7.66 ms**；当前 `aic_cube≈4%`，I4 暂缓。下一主刀为 **I5 窗 soft-lead Prefill**（对标 isub `VEC_2WIN_PIPE`，非照搬 PR190 Process——PR190 无 Prefill）。细节与门禁数字见 [ITER_LOG.md](ITER_LOG.md)。
