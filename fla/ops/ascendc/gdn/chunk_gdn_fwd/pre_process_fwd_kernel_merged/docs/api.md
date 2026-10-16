# pre_process_fwd_kernel_merged 接口与计算约定

本算子实现线性 Attention 在**上下文并行（CP, context parallel）**下的前处理步骤：为当前
rank 的**一个序列窗口（一个 part）**计算窗口边界状态 `h` 与仿射链矩阵 `m`，并按上游布局
打包写回 `hm`。上游 `merge_fwd_bwd_kernel` 负责在 rank 间合并，不由本算子承担。

工作流：`catlass-linear-attention-v1`（CANNBot Linear Attention 专用流程，五阶段：
01 接口确认 → 02 标杆生成 → 03 方案设计 → 04 算子开发 → 05 算子测试）。

## 1. 参考资料与固定版本

| 项目 | 内容 |
| --- | --- |
| 参考实现仓库 | `fla-org/flash-linear-attention`（GitHub 上游） |
| 固定 commit | `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`（`main`，2026-09-17 读取） |
| 参考文件 | `fla/ops/cp/chunk_delta_h.py`，`pre_process_fwd_kernel_merged`（第 42 行起） |
| 参考文件 SHA256 | `a6ed6aaaf0bc6c7dc8c9235a5a9346d14a0bbeb3630b8baf8f1c2bc7ac92435c` |
| 许可证 | MIT（上游仓库根目录 `LICENSE`） |
| 本地归档 | `/home/npu_user7/BartonFang/refs/fla_pinned/`（含 `PIN.txt` 与逐文件 SHA256） |
| 用户需求来源 | 用户指定该 kernel 作为本 AscendC 算子的对标接口 |

同一 commit 下的调用方（决定各分支的实际语义）：

| 调用方 | 传参特征 | 生效分支 |
| --- | --- | --- |
| KDA `fla/ops/kda/chunk_fwd.py` | `k=kg, w=w, u=u, gk=g, v=None` | `USE_GK`，`v` 复用 `u` |
| GDN `fla/ops/gated_delta_rule/chunk.py` | `k=k, w=w, u=u, g=g, v=None` | `USE_G`，`v` 复用 `u`，`HK` 可小于 `HV` |
| DPLR `fla/ops/generalized_delta_rule/dplr/chunk.py` | `k=kg, w=w, u=u, gk=gi, bg=bg, v=v` | `USE_GK` + `USE_BG`，`v`/`u` 为不同张量 |

上游 Python 包装的窗口切分：`cp_context.layout == 'zigzag'` 时按 `front`/`back` 两个 part
分别取 `cu_seqlens[fns-1:fns+1]` 与 `cu_seqlens[-2:]`，对每个 part 调用一次本 kernel
（`is_last_by_part` 为真时跳过）。**每次 kernel 调用处理一个窗口**，这正是本算子的粒度。

## 2. 本次范围

**做**：单窗口、单 part，给定该窗口的 `k/v/u/w/g/gk/bg/cu_seqlens`，输出完整 `hm`。
算法族覆盖 GDN（`g`）、KDA（`gk`）、DPLR（`gk` + `bg`）三条路径。

**不做**（由框架侧或后续版本承担）：

- CP 多卡编排：`all_gather_into_tensor`、`merge_fwd_bwd_kernel`、zigzag 的 part 选择与
  `is_last_by_part` 判定；
- `MULTI_SEQS` 路径（上游包装恒传 `False`）；
- `state_v_first`（它只影响包装里 `initial_state` 的物理布局，不影响本 kernel 的 `hm` 输出）；
- `AFFINE_CHAIN_PRECISION = tf32x3`（NVIDIA 专用；NPU 固定 `ieee`，即 FP32 累加）。

## 3. 接口

### 3.1 算子名与层级

| 项目 | 取值 |
| --- | --- |
| 算子名（snake） | `pre_process_fwd_kernel_merged` |
| OP TYPE | `PreProcessFwdKernelMerged` |
| aclnn 接口 | `aclnnPreProcessFwdKernelMerged` |
| 工程目录 | `fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/` |
| Python 调用 | `from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged`（同时提供 `npu_pre_process_fwd_kernel_merged`） |
| 接入方式 | 仓内 AscendC 算子 + `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` ctypes wrapper + `ops/ascendc/__init__.py` 的 `_ASCENDC_OPS` 注册 |
| 目标 SoC | Ascend950（`NpuArch=3510`，`CATLASS_ARCH=3510`） |

`_ASCENDC_OPS` 注册时会同时导出 `npu_pre_process_fwd_kernel_merged` 与去掉前缀的
`pre_process_fwd_kernel_merged`，与 `npu_chunk_fwd_h` / `chunk_fwd_h` 的现有约定一致。

### 3.2 输入

`B` 为窗口条数（CP 场景下每个 rank 的每个 part 对应一条，通常 `B = 1`）；`T` 为窗口长度；
`K` 为 key 维，`V` 为 value 维，`HK` 为 key 侧 head 数，`HV` 为 value 侧 head 数。
布局沿用本仓 AscendC 算子约定 `[B, H, T, D]`（BNSD），与 `chunk_fwd_h` 一致。

| 名称 | 必选 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| `k` | 是 | `[B, HK, T, K]`；gk 模式下 `HK == HV` | BF16 | g-only 路径为 raw key；gk 路径为已完成 gate 准备的 `kg`，本算子不再乘 gate |
| `w` | 是 | GDN/KDA：`[B, HV, T, K]`；DPLR：`[B, HK, T, K]` | BF16 | erase/WY 输出，h 与 m 的左矩阵 |
| `u` | 是 | `[B, HV, T, V]` | BF16 | GDN/KDA 下与 `v` 为同一张量（仅在 DPLR 分支被读取） |
| `v` | 是 | `[B, HV, T, V]` | BF16 | DPLR 下独立于 `u`；GDN/KDA 下等于 `u` |
| `g` | 二选一 | `[B, HV, T]` | FP32 或 BF16 | 标量 gate，**base-2 的 chunk 内累积对数衰减**；与 `gk` 互斥 |
| `gk` | 二选一 | `[B, HV, T, K]` | FP32 或 BF16 | 逐 K gate，同为 base-2 chunk 内累积量；与 `g` 互斥 |
| `bg` | 仅 DPLR | `[B, HK, T, K]` | BF16 | DPLR 的 K 侧项，同时作为 h 的第二个左矩阵与 m 的 K 矩阵 |
| `cu_seqlens` | 否 | `[N+1]` | INT32 或 INT64 | 变长模式；本算子只消费当前窗口的起止（`N = 1`） |

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `64` | 本轮固定 `64`，其它值在 host 侧拒绝 |
| `use_bg` | bool | `false` | 由 `bg` 是否存在推导；仅 DPLR 为真 |
| `AFFINE_CHAIN_PRECISION` | - | `ieee` | 固定 FP32 累加；`tf32x3` 不支持 |

### 3.4 输出

| 名称 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `hm` | `[B, HV, K, V+K]` | FP32 | 单个 head 的 `[K, V+K]` 矩阵：左 `[0, V)` 为 `h`，右 `[V, V+K)` 为 `m` |

上游布局为 `(2, HV, K, V+K)`，第 0 维是 part（front/back）。本算子一次只产生一个 part，
由调用方决定写入哪一片；本算子不改变单 part 内部 `[HV, K, V+K]` 的排布与 `K+V` 步长。

## 4. 数学语义

按一个 value head `i_h`、一个窗口描述，`T` 为该窗口 token 数，`BT = 64`，
`NT = ceil(T / BT)`。累加一律 FP32。`E(x) = exp2(x)`（**全部衰减量都是 base-2 对数域**）。

状态：`h` 为 `K x V`，初值全零；`m` 为 `K x K`，初值单位阵。

对第 `c` 个 chunk（`c = 0 .. NT-1`），记 `last = min((c+1)*BT, T) - 1`：

```text
# 1) 用上一 chunk 遗留的 h（未按本 chunk 衰减）算 v 的衰减项，dot 前 h 先降到 BF16
v_decay = W_c @ bf16(h)

# 2) 形成本 chunk 的“新 v”
GDN/KDA:  v_new = V_c - v_decay
DPLR:     v2    = v_decay + U_c ;  v_new = v2

# 3) chunk 内逐 token 衰减
USE_G:    v_new *= E(g[last] - g[t])   # 逐行，t 为 chunk 内全局 token 下标
          h     *= E(g[last])          # 整个状态按本 chunk 的总衰减缩放
USE_GK:   h[k, :] *= E(gk[last, k])    # 逐 K 行衰减（此时 v_new 不再单独缩放）

# 4) 累积本 chunk 的贡献（右矩阵降为 BF16 后做 Cube 乘累加）
GDN/KDA:  h += K_c^T @ bf16(v_new)
DPLR:     h += K_c^T @ bf16(V_c) + bg_c^T @ bf16(v_new)

# 5) 仿射链 m 的推进（全 FP32）
USE_G:    K_c = K_c * E(g[last] - g[t])[:, None]
          M_c = diag(E(g[last])) - K_c^T @ W_c
USE_GK:   M_c = diag(E(gk[last, :])) - K_c^T @ W_c
DPLR:     M_c = diag(...)            + bg_c^T @ W_c     # 注意为 “+”
m = M_c @ m        # 按 chunk 顺序左乘，m 初值为 I
```

其中 `K_c` 在 DPLR 下即 `bg_c`，在其余分支为 `k_c`（g-only 时 `k_c` 复用的是 `HV` 侧展开后的
key head `i_h // (HV / HK)`）。

写回：

```text
hm[i_h, 0:K, 0:V]     = fp32(h)
hm[i_h, 0:K, V:V+K]   = m
```

**必须保持的舍入点**（与上游逐位对齐的关键）：

1. `v_decay` 的 dot 中 `h` **先降到 BF16 再入 dot**；
2. `h` 的累积 `h += K^T @ bf16(v_new)` 在 FP32 累加器上进行，不被中途舍入；
3. `m` 的整条链在 FP32 内完成，不在 chunk 之间降精度；
4. `hm` 为 FP32 输出，不做 BF16 舍入。

## 5. 支持矩阵（本轮）

| 维度 | 范围 |
| --- | --- |
| 算法族 | GDN（`g`）、KDA（`gk`）、DPLR（`gk` + `bg`）全覆盖 |
| `K` | `K <= 256`；`h` 的 K 维按 64 行分块（最多 4 块），`BLOCK_SIZE` 取两档：`K <= 64` 用 32，否则用 64 |
| `V` | 与 `K` 同量级，按 `BLOCK_SIZE` 列切分 |
| `BT` | 固定 64 |
| `B` | 窗口条数；本轮 CP 用例以 `B = 1` 为主，`B > 1` 是否纳入验收由 03 阶段按 tiling 确定 |
| 序列 | 定长（`cu_seqlens` 缺省）与变长（`cu_seqlens` 存在），尾块与不满一个 chunk 的尾部 |
| dtype | `k/w/u/v/bg` BF16；`g/gk` FP32 或 BF16；`hm` FP32 |
| SoC | Ascend950 / `NpuArch=3510` |

## 6. 边界与异常

- 空窗口（`T = 0`）：host 侧拒绝，返回非零错误码。
- `K > 256`：host 侧拒绝。
- `g` 与 `gk` 同时提供或同时缺失：host 侧拒绝。
- 提供 `bg` 但未提供 `gk`（或反之）：DPLR 要求两者同时存在，host 侧拒绝。
- 变长模式下 `N != 1`：本轮拒绝（单窗口粒度）。
- `u` 与 `v` 非同一张量且未提供 `bg`：host 侧拒绝（GDN/KDA 规定 `v` 复用 `u`）。
- K 或 V 非 `BLOCK_SIZE` 整数倍：允许，由尾块掩码处理，`hm` 越界区域不写。
- `hm` 的非有效区域（`k >= K` 或列越界）不写、不参与比较。

## 7. 性能目标

**对标 1.0 倍 H20**：在约定的模型 case 上，本算子的 kernel 耗时必须达到 H20 上同一
`pre_process_fwd_kernel_merged` kernel 的 1.0 倍以内（即 `T_npu <= 1.0 x T_h20`）。

| 项目 | 取值 |
| --- | --- |
| 对标对象 | NVIDIA H20 上运行的上游 Triton kernel `pre_process_fwd_kernel_merged` |
| 目标倍率 | 1.0x |
| 模型 case（shape / dtype / SoC） | **待用户提供**（见 §8） |
| H20 基线数值与采集口径 | **待用户提供**（预热次数、采样次数、统计量、是否含上下游 kernel） |
| 判定指标 | `msprof` `op_summary` 的 `Task Duration(us)`，与本算子自身基线同条件比较 |

本机没有 NVIDIA 设备，H20 基线无法在本地复现，只能采用用户提供的实测数据；因此该 case 的
shape、采集口径与基线数值必须在 03 阶段进入设计前固定下来。

## 8. 尚未解决、需要用户选择的问题

1. **H20 基线数据**：需要给出对标模型 case 的完整 shape（`B / HK / HV / T / K / V /
   chunk_size`、dtype、是否 varlen）、H20 上的实测耗时，以及采集口径（预热/采样次数、
   统计量、是否只统计该 kernel）。这是 03 阶段性能设计的输入，缺失时 `stage` 不能进入 03。
2. **V 维是否合并**：上游同一个 kernel 用同一 grid 的不同 program 同时产出 h（V 列）与
   m（K 列），本算子是否照做（一个 kernel 内按列分块产出完整 `hm`），还是拆成两个
   Stage / 两个 kernel 再拼装。**用户要求再讨论，暂不决定**；该选择直接决定 Stage 划分、
   workspace 布局与核间同步方案，需在 03 阶段开始前定稿。
3. **命名门禁偏离**：本流程的 Step 0/Step 1 要求算子名包含 `catlass`，而用户指定的
   `pre_process_fwd_kernel_merged` 不含该子串（为与上游 kernel 对齐而有意偏离）。已按用户
   明确要求执行，`select_operator_workflow.py` 的 `legacy` 分支判断在新建目录时为
   `route=pending`，不构成阻塞，但该偏离需在交付说明中记录。
4. **`g`/`gk` 数值域**：上游无条件使用 `exp2`（base-2 chunk 内累积）；本仓 `chunk_fwd_h` 有
   `use_exp2` 开关、默认自然对数。本算子暂固定 base-2（与上游一致），请确认是否需要在
   Python 适配层做换算。
5. **`u`/`v` 接口形态**：保留两个入参并在 host 校验别名（本文件采用的方案），还是对
   GDN/KDA 直接收敛为单一入参。
6. **测试落点**：本流程默认在算子工程内用 `test/` + `scripts/compare_precision.py`；本仓另有
   `tests/atk/<op>/` 的 ATK 验收工程。需要在 02/05 阶段确定两套入口的分工（建议：开发期用
   流程自带入口，最终验收补 `tests/atk/pre_process_fwd_kernel_merged/`）。