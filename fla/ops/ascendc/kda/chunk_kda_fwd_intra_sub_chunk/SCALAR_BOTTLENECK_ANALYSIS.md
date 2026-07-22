# Intra Sub Chunk：AIV Scalar 高耗时根因与改进方案

> 基于 msprof `op_summary`（模型 `(1,32,8192,128)` BT=64，墙钟 ~52 ms，`aiv_scalar_ratio≈0.43`）
>
> + 本算子 kernel 实现 vs `chunk_kda_fwd` / `solve_tri` 对比。
>   改进执行计划：`/root/.cursor/plans/intra_sub_chunk_post_vectorize_9c2e4a1b.plan.md`

---

## 1. Profiling 事实（再陈述）

| 指标                 | 值                        | 含义                   |
| -------------------- | ------------------------- | ---------------------- |
| Task Duration        | ~52.3 ms                  | 墙钟                   |
| `aiv_scalar_ratio` | **0.426**（~22 ms） | AIV 标量管线占主导     |
| `aiv_vec_ratio`    | 0.09（~4.6 ms）           | 向量 prep 已部分生效   |
| `aic_mac_ratio`    | 0.001（~72 µs）          | Cube GEMM 几乎不占墙钟 |
| `aic_mte2_ratio`   | 0.005                     | 搬入可忽略             |

**结论：** Cube 已把 score GEMM 从热路径拿掉；墙钟由 **AIV0 的 `PostSub` 标量环**决定。AIC 大量空等 AIV。

---

## 2. 本算子 Post 路径解剖

Cube 路径 CV 循环（简化）：

```text
for i_sub:
  AIV0+AIV1: PrepareSub (向量 Cast/Exp/Mul + DataCopy)   ← 相对健康
  AIC:       ComputeMmad ×2                              ← ~µs 级
  AIV0 only: PostSub                                     ← 标量灾难
  AIV barrier
```

`PostSub`（`chunk_kda_fwd_intra_sub_chunk.cpp` ~892–971）四段：

| 段 | 代码                      | 实现方式                                 | 复杂度 / 每 sub                             |
| -- | ------------------------- | ---------------------------------------- | ------------------------------------------- |
| A  | 读 cmat                   | `DataCopy`                             | OK                                          |
| B  | tril + scale + β→`-L` | **双重 `GetValue`/`SetValue`** | O(BC²)=256 次标量                          |
| C  | forward-sub`(I-L)⁻¹`  | **三重标量环**                     | O(BC³)≈~2k MAD                            |
| D  | store Aqk/Akkd            | **逐元素 Cast+SetValue**           | O(BC²)，且 Aqk 每元素`SyncSV`+`SyncVS` |

### 2.1 最毒：`StoreAqkRow`

```870:882:fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp
    __aicore__ inline void StoreAqkRow(...)
    {
        ...
        for (uint32_t j = 0; j < static_cast<uint32_t>(bc_); ++j) {
            sf.SetValue(0, row.GetValue(j));
            SyncSV();
            Cast(st, sf, RoundMode::CAST_RINT, 1);
            SyncVS();
            aqk_.SetValue(base + j, st.GetValue(0));
        }
    }
```

每个元素：**1 次标量搬 + 2 次硬同步 + 1 元素 Cast + 1 次 GM SetValue**。
BC=16 → 每行 16 次；每 sub 16 行 → **256 次 SyncSV/VS 对**。

模型量级估算（每 AIC 核约 `4096/20≈205` chunk × `NC=4`）：

```text
StoreAqk Sync 对 / 核 ≈ 205 × 4 × 256 ≈ 2.1e5
```

这会直接堆在 `aiv_scalar_time` 上；`StoreAkkdRow` 虽无 Cast sync，但仍是 256 次 `SetValue`/sub。

### 2.2 forward-sub 标量三重环

```937:954:fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp
        for (uint64_t i = 2; i < valid; ++i) {
            ...
            for (uint64_t j = 0; j < i; ++j) {
                float acc = tmp.GetValue(...);
                for (uint64_t p = 0; p < i; ++p) {
                    acc += tmp.GetValue(p) * akk.GetValue(p * bc_ + j);
                }
                ...
            }
        }
```

经典 `(I-L)⁻¹` 前代换，**全部走 scalar unit**，无向量 MAD。BC=16 时绝对 flop 不大，但每 flop 经 `GetValue`/`SetValue`，吞吐极低。

### 2.3 tril/β 同样是逐元素分支

```927:935:.../chunk_kda_fwd_intra_sub_chunk.cpp
        for (i,j in BC×BC) {
            sa = aqk[i,j]*scale; sk = akk[i,j]*beta[i];
            aqk = (tril?) sa : 0;
            akk = (strict_tril?) -sk : 0;
        }
```

本可用 `Muls` + `Select`/`mask` 整块做完。

### 2.4 结构放大器：仅 AIV0 post + 严格 barrier

```988:993:.../chunk_kda_fwd_intra_sub_chunk.cpp
            if (subBlockIdx == 0) {
                PostSub(...);
            }
            Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
```

- AIV1 在整个 Post 期间空转
- prep(i+1) 不能与 post(i) 重叠（DEPTH=2 槽位已备但未用）
  → 墙钟 ≈ `prep + wait_AIC + post_AIV0`，post 越慢整体越慢

### 2.5 对比：同文件 Prep 已向量化

`PrepareSub` 用 `Cast`/`Exp`/`Mul` + `DataCopy` 写 scratch，**禁止 SetValue→DataCopy**（注释硬性）。
**Prep 健康、Post 仍是 scalar fallback 拷贝** → profile 上 `aiv_vec` 有一点、`aiv_scalar` 爆炸，与代码结构一致。

---

## 3. 与其它算子 Kernel 对比

| 算子                           | 三角/前代换做法                                                                           | Store                              | GetValue/SetValue 热路径 |
| ------------------------------ | ----------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------ |
| **本算子 PostSub**       | O(BC³) 标量前代换                                                                        | 逐元素 SetValue + 单元素 Cast sync | **极密**           |
| **`chunk_kda_fwd`**    | Neumann 倍增（MCH）**16×16 对角块上 Cube MMAD** + 向量 `Select`/`Brcb`/`Mul` | `DataCopy`/`DataCopyPad` 整块  | **热路径 0**       |
| **`solve_tri`（GDN）** | 同 MCH，`SolveTriCube<16>`                                                              | 向量辅矩阵 + Cube                  | **热路径 0**       |
| **GDN prepare_wy_***     | 向量/Cube                                                                                 | DataCopy                           | **0**              |

### 3.1 `chunk_kda_fwd` 可直接对标的积木

| 本算子 Post 段 | 参考实现                                                                | 位置                              |
| -------------- | ----------------------------------------------------------------------- | --------------------------------- |
| tril mask      | `BuildCausalSelectMasks` + `SelectCausalRows`                       | `chunk_kda_fwd.cpp` ~1103–1146 |
| β 行缩放      | `Brcb` + `Mul`                                                      | ~1159–1171                       |
| `(I-L)⁻¹`  | `ComputeAkkInverseMchFull`，`KDA_SOLVE_DIAG_BT=16`，`MCH_ITERS=3` | ~1593–1632                       |
| 写回           | `DataCopy` / `StoreFloatRow`（整行 Cast+Copy）                      | ~487–496, 1392+                  |

要点：`chunk_kda_fwd` 的对角块大小 **就是 16**，与本算子 `BC=16` **同构**。不是「大矩阵才值得上 Cube」——参考算子已经在 16×16 上用 Cube 做逆，本算子却用标量前代换，是历史债务，不是数学必然。

### 3.2 数学等价

严格下三角幂零 `L`：

```text
X = (I - L)^{-1} = I + L + L² + … + L^{BC-1}
```

MCH 倍增：`Y←L²` 迭代、`X←X+X·Y`，`⌈log2(BC)⌉` 轮 → BC=16 时 **3 轮 16×16 Cube GEMM**，与标量前代换数值目标一致（fp32）。

---

## 4. 耗时归因拆分（定性 → 可测）

无法从聚合 `aiv_scalar_ratio` 再拆子函数，但按实现成本排序：

| 优先级 | 段                        | 为何贵                         | 预期占 scalar 份额（估）   |
| ------ | ------------------------- | ------------------------------ | -------------------------- |
| P0     | `StoreAqkRow`           | 每元素 2×Sync + 标量 Cast/GM  | **很大（常是第一）** |
| P0     | `StoreAkkdRow`          | 每元素 GM SetValue             | 中                         |
| P1     | forward-sub 三重环        | O(BC³)×Get/Set               | 大                         |
| P1     | tril/β 双重环            | O(BC²)×分支 Get/Set          | 中                         |
| P2     | AIV1 空闲 + 无 prep‖post | 放大墙钟，不直接加 scalar 计数 | 墙钟 10–30% 级            |

验证法：先只改 store → 复采 `op_summary`，看 `aiv_scalar` / Duration 下降幅度，再改 solve。

---

## 5. 改进方案（分层）

### Phase A — 向量化 Store（低风险、快）

**目标：** 消灭逐元素 `SetValue` / 单元素 Cast sync。

| 输出                             | 做法                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `akkd` `[..., BC]`           | UB 上整行/整块`DataCopy` → GM（fp32 对齐）                                                |
| `aqk` `[..., BT]` 的 BC 列块 | UB：`Cast` 整行 BC 个 fp32→T，再 `DataCopy`/`DataCopyPad` 到 `AqkOff(..., iSub*BC)` |

对标：本算子 `PrepareSub` 的 `CopyVectorOut`；`chunk_kda_fwd::StoreFloatRow`。

**验收：** smoke + 模型采样绿；`aiv_scalar_ratio` 明显下降；Duration 目标先砍一截。

### Phase B — 向量化 tril / β（中风险）

1. `Muls(aqk, aqk, scale, BC*BC)`
2. `Brcb(beta)` + 行 `Mul` → akk
3. `Muls(akk, akk, -1)`
4. 因果 mask：`Select` + bitmask（缩版 `SelectCausalRows`，BT→BC）
5. 对角 `+I`：`Adds` 或 mask 加 1

**仍可留在 AIV**；为 Phase C 准备干净的 `L` / `X0`。

### Phase C — 16×16 逆：Cube MCH 替换标量前代换（高收益）

**推荐主路径（与参考算子对齐）：**

```text
AIV: 准备 X0=I-L、Y0=L²（或直接写 GM scratch）→ flag
AIC: 3× CubeGemm 16×16×16（对标 ComputeAkkInverseMchFull 对角块）
AIV: DataCopy 写 akkd
```

备选：复用/抽出 `solve_tri` 的 `SolveTriCube<16>`，避免再抄一套 MCH。

**不推荐：** 继续标量前代换但「手写向量内积」——BC=16 仍 O(BC³) 且难维护，收益不如直接抄已验证的 MCH。

Workspace：DEPTH 槽上增加小块 solve scratch（或复用 cmat 平面），注意与 score ping-pong 不打架。

### Phase D — 流水与并行（墙钟）

1. **双 AIV 拆 post**：行划分 tril/scale/store；solve 若走 Cube 则 AIV 只做准备/写回
2. **prep(i+1) ‖ post(i)**：DEPTH=2；AIV1 prep next / AIV0 post cur（或 post 向量化后双 AIV 再叠）
3. 严格保证 flag 成对 + 不写同一 GM 行竞态

### Phase E — 明确不做 / 后置

- 再抠 score Catlass / kneg L1（profile 已证明非瓶颈）
- fp32 MMAD score（精度口径已否决）
- 改 ABI / BC

---

## 6. 预期收益（模型 shape）

| 阶段 | 实测 / 预期 | 依据 |
|------|-------------|------|
| A Store 向量化 | **实测** 52.3→36.0 ms | Sync 环消失；`aiv_mte3` 上升 |
| B tril/β 向量化 | **实测** 再降到 **30.2 ms** | 行向量 Mul/Muls + prefix mask |
| C Cube MCH | **实测** 再降到 **18.0 ms** | 替换 O(BC³) scalar；`aic_mac` 72→266 µs |
| D 流水 | 中（藏 prep/post） | 当前 AIV1 空闲 |

已完成 A+B+C（约 **−66%**）。下一刀可选 Phase D。

### 6.1 Phase A+B 实现要点（已合入）

- `StoreAqkRow`/`StoreAkkdRow`：整行 `Cast_RINT` + `CopyVectorOut`；Post 末 `SyncSV`
- `LoadBetaRows`：对齐基址 `Duplicate(0)` + `Cast`；禁止 `Duplicate(beta[valid],…)`（UB 非对齐）
- `ApplyTrilScaleBeta`：`Muls(scale)` → 行 `Muls(β)` → prefix `Mul` mask → `Muls(-1)`

---

## 7. 风险与精度

| 风险                          | 缓解                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| Store Cast 模式与现口径不一致 | 保持`CAST_RINT`，整行 Cast 后 DataCopy                     |
| MCH vs 前代换数值差           | 对标`chunk_kda_fwd` / golden `score_dtype`；fp32 scratch |
| CV flag / DEPTH 槽冲突        | 单测 NC=4 + DEPTH=2；空 sub 仍成对 flag                      |
| 双 AIV post 写冲突            | 先 A 单 AIV0 DataCopy；再按行拆                              |

---

## 8. 参考路径

- 本 kernel：`op_kernel/chunk_kda_fwd_intra_sub_chunk.cpp`（`PostSub` / `Store*`）
- 参考：`kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp`（`SelectCausalRows` / `ComputeAkkInverseMchFull`）
- 参考：`gdn/chunk_gdn_fwd/solve_tri/`（`SolveTriCube<16>`）
- Prof：`prof_intra_sub_chunk_model*/**/op_summary_*.csv`
- 分核背景：`PARTITION_CUBE_ANALYSIS.md`
