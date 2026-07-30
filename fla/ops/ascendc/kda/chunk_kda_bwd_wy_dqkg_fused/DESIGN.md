# ChunkKdaBwdWyDqkgFused — 冻结契约

对标 Triton `chunk_kda_bwd_kernel_wy_dqkg_fused`。本文是实现冻结边界；Stage 细节见仓库外方案，以下为可编译契约。

## 1. 算子边界

- 独立 L0，`KERNEL_TYPE_MIX_AIC_1_2`
- 融合：状态路 `dq/dk/dw/dg*` + WY 路 `dv2/db/dA`（`dw` 不外露）
- chunk 间 **无依赖**；按 `B*NT`（varlen：`len(chunk_indices)`）分核；`HV` 核内串行

## 2. Layout / dtype

| 张量 | BNSD shape | dtype |
|------|------------|-------|
| q, k | `[B,H,T,K]` | bf16/fp16 |
| v, v_new, do, dv | `[B,HV,T,V]` | bf16/fp16 |
| g | `[B,HV,T,K]` | bf16/fp16（内部 exp2 用 fp32） |
| beta | `[B,HV,T]` | 同 q |
| A (Akk) | `[B,HV,T,BT]` | 同 q |
| h, dh | `[B,HV,NT,K,V]` 或 `[…,V,K]` | 同 q；由 `state_v_first` 决定末两维 |
| dq, dk, dg | `[B,HV,T,K]` | **fp32** |
| dv2 | `[B,HV,T,V]` | 同 q |
| db | `[B,HV,T]` | **fp32** |
| dA | `[B,HV,T,BT]` | **fp32** |

- SOC：ascend910b / ascend910_93 / ascend950
- `BT∈{64}` 首发；`K=128`；`V∈{128,256}`
- GVA：输出 HV 维，调用方再 reduce 到 H（与 Triton 一致）
- varlen：`cu_seqlens` + `chunk_indices` 成对；`B=1`

## 3. Attr

- `scale` float
- `chunk_size` int64
- `state_v_first` bool（false：`h` 为 `[…,K,V]`；true：`[…,V,K]`）

## 4. Golden

- Cube-faithful：GEMM 前 cast 到 qk dtype；固定 `exp2`
- 禁止收窄 range / 删大 T / 放宽阈值过门
- 模型 case：`B1 T8192 H=HV=32 K128 V128 BT64 bf16`，`state_v_first=false`
- 性能：Task Dur med ≤ **0.8 ms**（裸 msprof，筛 MIX_AIC）

## 5. Stage（修订版）

| Stage | 粒度 | 要点 |
|-------|------|------|
| 0 WyV | 每 head | Cube `dv@v.T`+`A@dv` → Vec `dv2/db` |
| 1 KvAcc | 每 BK | Cube `dq/dk/dw` ∥ Vec 造 `kg`（互不等） |
| 2 GateWy | 每 BK | Vec gate → Cube `dA/dkgb` → Vec epilog |
| 3 DaFinal | 每 head | Vec mask → Cube `dA@A`×2 → Vec store |

Window：2-head 语义下 **4 GM slot**；raw CrossCore `0x2`。

## 5.1 调度权威（Cube / Vec 必须镜像）

```text
windowStartSlot = (windowIdx & 1) * 2   # {0,2}
window 内 head0 → slot^0, head1 → slot^1
windowIdx 跨 task 连续递增（勿每 task 清零）

for task:
  for hvBase in 0..HV step 2:
    # Stage0 成组
    for h: Cube RunStage0; Set C_S0
    for h: Vec Wait C_S0; Stage0Vec; Set V_S0
    for h: Cube Wait V_S0
    for iK:
      for h: Cube RunStage1; Set C_S1          # Vec Kg 与此并行，无互 wait
      for h: Vec KgVec                        # 无 Wait C_S1
      for h: Vec Wait C_S1; GateOnlyVec; Set V_GATE
      for h: Cube Wait V_GATE; RunStage2; Set C_S2
      for h: Vec Wait C_S2; EpilogVec
    for h: Vec Mask; Set V_MASK
    for h: Cube Wait V_MASK; RunStage3; Set C_S3
    for h: Vec Wait C_S3; Store; Set slotFree
    ++windowIdx
```

| Flag | 值 | 方向 | 语义 |
|------|----|------|------|
| C_S0 | 0 | Cube→Vec | Stage0 dA/dvb ready |
| C_S1 | 1 | Cube→Vec | Stage1 dq/dk/dw ready |
| V_GATE | 2 | Vec→Cube | gate dwNeg/kg ready |
| C_S2 | 3 | Cube→Vec | Stage2 dADelta/dkgb ready |
| V_MASK | 4 | Vec→Cube | masked dA ready |
| C_S3 | 5 | Cube→Vec | dA@A@A ready |
| V_S0 | 6 | Vec→Cube | Stage0Vec done（进 Stage1 前） |
| SLOT_FREE0..3 | 7,11,12,13 | Vec→Cube | slot 可复用（跳过 Catlass 保留 8/9/10） |

`SetV*Joined` = `PipeBarrier(MTE3)+Set(0x2)`（**无** AIV↔AIV `Barrier(0x1)`；双方各自 Set，Cube Wait 收 2 信用）。双 AIV 按连续半块分活；`db`/`dgk` 经 `dbMergeWs`/`dgkMergeWs` 归约。

## 5.2 已知反模式（禁止回退）

1. **单 head 全流水**：`for iHv` 内跑完 Stage0–3 再下一 head（无法 stage 成组重叠）
2. **kg 挂在 Wait(C_S1) 之后**：消灭 Stage1 Cube∥kg，Cube 在 `C_S1→V_GATE` 空等
3. **双 AIV 同算同写**：第二颗 AIV 零有效吞吐；共享 GM 双写破坏精度

## 5.3 验收口径

| 项 | 标准 |
|----|------|
| 精度 | `test_npu_chunk_kda_bwd_wy_dqkg_fused.py` 全绿（含 varlen / `state_v_first`） |
| 调度/仿真 | PEM：同 window 两 head 的 `C_S0/C_S1` 成对；Kg 活动早于 Wait S1；`aiv0/aiv1` 行地址交错 |
| 性能 | 板端 MIX_AIC Task Dur med；冲 **≤0.8 ms**（不用 host `avg=`） |

## 6. 流水开关（kernel 宏）

| 宏 | 默认 | 含义 |
|----|------|------|
| `USE_L1_A_RESIDENT` | **1** | Stage0 多 V-tile 复用 L1 中的 `A`（`skipLoadA`） |
| `USE_STAGE1_L0C_ACCUM` | **1** | Stage1 跨 V-tile L0C 累加，一次 Fix→plane0；Gate 单平面读 |
| `USE_L0_AB_DBUF` | 0 | L0A/B ping-pong |
| `USE_FIX_MTE2_OVERLAP` | 0 | Fix∥下一 tile MTE2（`DirectTileGemmPipeState`） |

精度路径：I1/I2 已门禁 default on（suite PASS；model Task Dur ~**21.5 ms** vs ~22.0 ms 基线，Δ≤−0.05）。硬目标 0.8 ms 以板端 med 为准。

## 7. Ascend950 / arch35

- 入口：`__CCE_AICORE__ == 310` → `op_kernel/arch35/*`；否则 910B 父目录源码。
- Cube：复用父实现（`CATLASS_ARCH=3510` / `Ascend950` + Fixpipe `CopyL0CToDst`）。
- Vector：**当前复用父目录 classic AscendC**（arch35 薄 include）。MicroAPI regbase 曾在板端触发 AICore **507015**，`regbase.h` 保留待掩码 Load 验证后再开。
- Host tiling 共用；CMake 在 `ascend950` 下加 `Ascend950PR_9599`。
- 验收：950 编译通过 + 板端精度 suite；性能另记。详见 [`ASCEND950_TEST_GUIDE.md`](ASCEND950_TEST_GUIDE.md)。
