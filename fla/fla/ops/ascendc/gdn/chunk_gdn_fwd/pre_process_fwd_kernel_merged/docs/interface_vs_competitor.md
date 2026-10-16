# pre_process_fwd_kernel_merged：本算子接口语义 vs 竞品（fla-org CP）

> 依据：本仓 `docs/api.md`（2026-09-23 版）+ 现网代码
> `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py::npu_pre_process_fwd_kernel_merged`
> + host tiling `op_host/pre_process_fwd_kernel_merged_tiling.cpp`
> + 竞品 `fla/ops/cp/chunk_delta_h.py`（commit `e52dbc0e`，`chunk_gated_delta_rule_fwd_h_pre_process`）。

---

## 1. 本算子的接口（当前实现的**权威语义**）

```python
from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as ppfm

hm = ppfm(k, w, u, g=..., gk=..., bg=..., v=..., cu_seqlens=[bos, eos, ...], chunk_size=64)
# hm: [Nseq, HV, K, V+K]  FP32（左 [0,V) = h，右 [V,V+K) = m）
```

| 项 | 取值 / 约束 |
|---|---|
| `k` | `[1, HK, T, K]` BF16；GDN 为 raw key，按 `hk = hv // (HV/HK)` 复用；KDA 为 `kg`（已完成 gate 准备，本算子不再乘 gate） |
| `w` | `[1, HV, T, K]` BF16 |
| `u` | `[1, HV, T, V]` BF16（GDN/KDA 下 `v = u`，`v=None` 即复用） |
| `v` | 仅 DPLR 需要显式给 |
| `g` / `gk` | **二选一**：`g [1,HV,T]`（GDN，标量 gate）/ `gk [1,HV,T,K]`（KDA，逐 K gate）；FP32 或 BF16；两者都是 **base-2 chunk 内累积对数衰减** |
| `bg` | 仅 DPLR（本轮预留、不验收） |
| `cu_seqlens` | **必给**，**host `list[int]`**（不是 device tensor），`[N+1]`，严格递增，`0 ≤ cu[0] < cu[-1] ≤ T`，**允许子区间** |
| `chunk_size` | 固定 `64`（其它值 host 拒绝） |
| `B` | **恒为 1**（varlen 打包）；`B>1` 由调用方打包进 token 轴 |
| `Nseq` | `len(cu_seqlens) - 1`（≥1），**不是入参** |
| `hm` | 返回值 `[Nseq, HV, K, V+K]` FP32（Python 侧预分配、kernel 写满） |
| 并行度 | `Nwork = Nseq × HV`（chunk 维不可并行，见 `design.md` 2.1.4） |
| 精度 | 输入 BF16、累加 FP32（native ieee）；**无 tf32x3** |
| head | `HV ≥ HK` 且 `HV % HK == 0`（GVA）；**KDA 的 `gk` 也允许 `HK < HV`**（gk 按 value head，k 仍按 HK 头映射）——代码已放开（wrapper 注释与 host tiling 都只校验 `Hv % Hk == 0`） |

要点：**一次调用 = 一个窗口**（窗口内可含多段），产出 **Nseq 份 `hm`**；
`Nseq = 1`（例如 `cu_seqlens=[bos, eos]`）时，去掉 size-1 维后与竞品单次调用的 `hm` **逐字节相同**。

---

## 2. 竞品的接口与调用形态（同一 commit 源码核对）

**kernel 签名（22 参）**：
`k, v, w, g, gk, bg, u, hm, cu_seqlens, T, H, HV, K, V, BT, BLOCK_SIZE, BK1, USE_G, USE_GK, USE_BG, IS_VARLEN, MULTI_SEQS, AFFINE_CHAIN_PRECISION`
其中 `hm` 是**预分配 buffer、kernel 原地写**；`cu_seqlens` 是 **device tensor**（kernel 内 `tl.load`）；
`IS_VARLEN` / `MULTI_SEQS` / `USE_*` / `BLOCK_SIZE` / `BK1` / `AFFINE_CHAIN_PRECISION` 都是 `tl.constexpr`。

**跨卡 CP 包装层**（`chunk_gated_delta_rule_fwd_h_pre_process`）：

| 布局 | 每次调用处理的段 | 调用次数 | `hm` | 跳过条件 |
|---|---|---|---|---|
| **contiguous**（默认） | 本 rank 窗口的**末段**：`cu_seqlens[-2:]` | 1 次 | `[HV, K, V+K]`（无前导维） | `is_last_rank`（graph 模式仍启动） |
| **zigzag** | part0 = front 的末段 `cu_seqlens[fns-1:fns+1]`；part1 = back 的末段 `cu_seqlens[-2:]` | **2 次** | 缓冲区 `[2, HV, K, V+K]`，每次写 `hm[part]` | 逐 part 的 `is_last_by_part` |
| 卡内切段（intracard） | **一次多段** | 1 次 | `[S_split, HV, K, V+K]` | — （`MULTI_SEQS=True`、grid 带 segment 维） |

调用后：`all_gather_into_tensor(hm, group)` → merge（`merge_fwd_bwd_kernel`）→ `initial_state` 记账，全在包装层。
**定长 `B>1`**：CP 路径不支持，调用方先把 `[B,H,T,D]` 打包成 `[1,H,B*T,D]` 并构造 `cu_seqlens`（等长 batch = 每段长度相同）。

---

## 3. 逐项对照（模型接入视角）

| 维度 | 竞品 | 我们 | 模型/框架要改什么 |
|---|---|---|---|
| 序列表达 | CP 恒 varlen：`B==1` + 打包轴上的 `cu_seqlens`（**device tensor**） | 同：`B≡1` + `cu_seqlens` 必给（**host `list[int]`**） | 打包逻辑可照抄；把 `cu_seqlens` 用 list 传（省一次 H2D；host 侧做边界校验与 tiling 展开） |
| 一次调用几段 | **1 段**（`MULTI_SEQS=False`） | **可 1..N 段** | 保守用法：传 2 元素 `cu_seqlens` → 与竞品 1:1；进阶用法：整窗一次算，再从 `hm[i]` 取所需段 |
| 输出 | `hm` 预分配、**原地写**；zigzag 前导维 = **part** | **返回** `hm[Nseq,HV,K,V+K]`；前导维 = **段** | `Nseq=1` 时逐字节相同；zigzag 想一次两 part：取 `hm[fns-1]` / `hm[-1]`（part 边界必在 `cu_seqlens` 里） |
| 返回值 vs buffer | 需要自己 `k.new_zeros(...)` 并复用 | Python 侧自动分配并返回 | 若框架要求写进 all_gather 的目标 buffer，需要加可选 `hm_out`（当前不支持，见 §5） |
| kernel 旋钮 | `T/H/HV/K/V/BT/BLOCK_SIZE/BK1/USE_*/IS_VARLEN/MULTI_SEQS/AFFINE_CHAIN_PRECISION` 全显式 | **都不暴露**：形状 + TilingKey 推导；`chunk_size=64`；精度固定 ieee | 调用方不需要传；要"照抄竞品签名"可加 shim（§5） |
| 定长 `B>1` | CP 路径不支持（调用方 pack；非 CP 直接 return） | 不支持（wrapper 强校验 `B==1`） | 必须先 pack 成 `[1,H,B*T,D]` + `cu_seqlens`（两边一致） |
| 子区间窗口 | 就是它的调用形态（`cu_seqlens[-2:]` 相对整根张量） | **明确支持**（host 放宽校验、kernel 按 `bos` 定位） | ✅ 无差异 |
| GVA（`HK<HV`） | GDN 支持；KDA 的 `gk` 按 value head，`k` 按 HK 映射 | 同（wrapper/host/kernel 均已放开） | ⚠️ 文档 `api.md` 仍写"gk 要求 HK==HV"，**属过期描述**，需修 |
| 精度口径 | 输入 BF16、fp32 累加；`AFFINE_CHAIN_PRECISION` 可选 `tf32x3` | 输入 BF16、fp32 累加（ieee） | 数值对标用竞品 `"ieee"` 档（H20 上是非 TF32 机器的默认支） |
| CP 编排（跳末段 / all_gather / merge / initial_state） | 在包装层，算子不感知 | **不在算子内** | 框架侧保留这一层；或用 `docs/integration_flow.md` 的等价写法（整窗多段 + 取段） |

---

## 4. 三种落地写法（按"贴近竞品程度"排序）

```python
# ① 最保守：与竞品逐行对应（每次一段，Nseq=1）
hm = ppfm(k_loc, w_loc, u_loc, gk=gk_loc, cu_seqlens=[bos, eos])   # e.g. [fns-1 : fns+1] / [-2:]
hm_no_lead = hm[0]            # == 竞品 kernel 写出的 [HV, K, V+K]（逐字节）

# ② 推荐：整窗一次算（吃满并行度；zigzag 两 part 也一次搞定）
hm = ppfm(k_loc, w_loc, u_loc, gk=gk_loc, cu_seqlens=local_cu)     # local_cu 含 part 边界
hm_front_last = hm[fns - 1]   # 竞品 part0 那一条
hm_back_last  = hm[-1]        # 竞品 part1 那一条

# ③ 定长 B>1：先打包（与竞品 CP 的 pack_bt 一致）
k_p = k.permute(1, 0, 2, 3).reshape(1, H, B * T, K)   # 其余同理；cu = [0, T, 2T, ..., B*T]
hm = ppfm(k_p, w_p, u_p, gk=gk_p, cu_seqlens=cu)      # hm[B, HV, K, V+K]：第 b 段 == 第 b 条序列
```

---

## 5. 待拍板 / 待办（接口层面）

1. **`api.md` 过期**：文中"gk 路径（KDA/DPLR）要求 `HK == HV`"（§3.2 表格与 §5 校验项）与现网实现
   （wrapper 注释 + host tiling 只校验 `Hv % Hk == 0`）不一致 —— 需按"KDA 允许 GVA（gk 按 value head）"改文档。
2. **是否要 `hm_out`**：若框架希望"结果直接写进 all_gather 的目标 buffer"（竞品是原地写），需要加一个可选
   输出参数；否则每次多一次小拷贝（`hm[Nseq,HV,K,V+K]`，模型 case 下 ~MB 级）。
3. **是否要"竞品同形 shim"**：在 Python 适配层加
   `multi_seqs: bool / is_varlen: bool / affine_chain_precision: str` → 映射成上面的 ①②③，
   使调用方可以**照抄竞品签名**而不用改算子契约（`api.md` §3.6 已预留这个方案）。
4. **DPLR**：接口已留（`bg` + 6 个 TilingKey），本轮不实现、不验收。
