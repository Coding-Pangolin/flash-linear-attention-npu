# pre_process_fwd_kernel_merged：触发条件、计算内容、结果去向

基于竞品 `fla-org/flash-linear-attention`（钉在 `e52dbc0e`）源码梳理，文件缩写：
`cp/chunk_delta_h.py` = `fla/ops/cp/chunk_delta_h.py`，`cp/context.py` = `fla/ops/cp/context.py`，
`common/chunk_delta_h.py` = `fla/ops/common/chunk_delta_h.py`。

## 0. 一句话

它是 CP（上下文并行）下的**跨卡状态接力棒**：把本 rank 的局部 token 窗口压缩成一次函数
`状态末 = h + m × 状态初`，跨卡把前缀的若干条链复合起来，得到"本 rank 窗口起点处应有的初始状态"，
交给下游 `chunk_gated_delta_rule_fwd_h` 当 `h0` 用。

## 0.1 术语：rank / part / 本地 buffer / 窗口 / 段 / 序列

| 术语 | 含义 | 谁决定 |
| --- | --- | --- |
| `total_tokens` | 打包轴总长（= 全局 `cu_seqlens[-1]`） | 数据 |
| `W` | CP world size（卡数） | 并行配置 |
| `num_parts` | `W`（contiguous）/ `2W`（zigzag） | `layout` |
| `part_len` | `total_tokens / num_parts`（**必须整除**，否则 `get_cp_cu_seqlens` 直接 raise） | `layout` |
| **part** | token 轴上**等长**的一段 `[k·part_len, (k+1)·part_len)`；切分单元（不是序列） | `layout` |
| `rank` | 一张卡；contiguous 持 **1 个** part，zigzag 持 **2 个** part（一头一尾） | 部署 |
| 本地 buffer | 该 rank 持有的 part 拼成的张量；zigzag 是 `[front; back]` | 调用方 |
| **窗口（window）** | 本算子一次调用看到的 T 轴 = 本地 buffer 整块（或某个 part 的子区间，若逐 part 调） | 调用方 |
| **段（segment）** | 窗口内**一截连续的序列**，边界由 `cu_seqlens` 给出 | 数据 + 切分 |
| 序列（sequence） | 全局原始序列；被 CP 切开后在各 rank 上各留一个段 | 数据 |

要点：**part 是"按 token 数等长切"的产物，与序列边界无关**，所以一个 part 里可能有多条序列的
片段，同一条序列也可能被 part 边界切成两段（见下面的 rank1 例子）。等长是硬要求：`all_gather`
要各 rank 形状一致；同时它保证各卡 token 数相同（负载均衡）。

**例（zigzag，`W=2`，`total=1024`，`segs=[0,40,600,700,1024]`——即实测的 `--preset multi`）**

```
num_parts = 2W = 4 ; part_len = 256
p0=[0,256)  p1=[256,512)  p2=[512,768)  p3=[768,1024)
rank0 ← p0 + p3   → 本地 buffer = 全局 [0,256) ∪ [768,1024)   （两段全局不相邻）
rank1 ← p1 + p2   → 本地 buffer = 全局 [256,768)              （相邻）

rank1 的本地 cu_seqlens = [0,256,344,444,512]（实测值）→ 4 个段：
  本地[0,256)   = 全局[256,512)  seq1（整段在 p1 内）
  本地[256,344) = 全局[512,600)  seq1 的尾巴（整段在 p2 内）← 同一条序列被 part 边界切成两段
  本地[344,444) = 全局[600,700)  seq2
  本地[444,512) = 全局[700,768)  seq3 的头
```

竞品对这部分要**调两次**：part `p1` 的末段 = 本地 `[0,256)`、part `p2` 的末段 = 本地 `[444,512)`，
分别写 `hm[0]` / `hm[1]`。我们整窗一次调用产出 `hm[0..3]`，取 `hm[0]`（front 末段）与 `hm[3]`
（back 末段）即可——**part 边界（本地 256 = 全局 512）本来就在 `cu_seqlens` 里**，所以段枚举天然
按 part 分开，不会把跨 part 的两段并成一段。

> zigzag 的存在理由（`cp/context.py` 的 docstring 原话）："Zigzag balances causal-attention CP
> layouts **without a re-shard at layer boundaries**"——即让每张卡同时持有一头一尾，避免跨层重排。

**两种切片写法是什么意思**（`cu_seqlens` 是**累计边界数组**，相邻两个元素 = 一段的 `[bos, eos)`）：

| 写法 | 取到的是 | 用途 |
| --- | --- | --- |
| `cu_seqlens[-2:]` | **本地轴上最后一段**的 `[bos, eos)` | contiguous：这个 part 的末段；zigzag：back part 的末段 |
| `cu_seqlens[fns-1 : fns+1]` | **front part 的末段**（`fns = front_num_seqs`） | 仅 zigzag：front part 的末段（它不是本地轴的最后一段，因为后面还有 back part） |

为什么正好切**两个元素**：kernel 在 `MULTI_SEQS=False` 下取 `i_n = 0`，只读
`cu_seqlens[0]`、`cu_seqlens[1]` 当 `bos/eos`——**"传两个元素"就是"这个窗口只含这一段"的表达方式**。
若把整条数组传进去而不切片，kernel 只会读它的**头一段**，那就取错段了。

用上例的 rank1（本地 `cu_seqlens=[0,256,344,444,512]`）对照：

```
cu_seqlens[-2:]        = [444, 512] → 本地最后一段 = 全局 [700,768)，即 back part 的末段
cu_seqlens[fns-1:fns+1] = cu[0:2]   = [0, 256]   → 本地第一段   = 全局 [256,512)，即 front part 的末段
```

对应到我们算子的输出：`hm[Nseq-1]` 就是 `cu_seqlens[-2:]` 那条链、`hm[fns-1]` 就是
`cu_seqlens[fns-1:fns+1]` 那条链——我们不需要写切片，因为"每段一条链"已经等价地把它们都产出了。

（`use_graph` 那段额外代码就是在防一个坑：末尾若有**零长 padding 段**，`cu_seqlens[-2:]` 会指向
padding 而不是真正有数据的最后一段，所以它改成从"非空段"里取 `i_last`。）

**同一段窗口、两种给法 → `Nseq` 不同**（这就是竞品 `MULTI_SEQS` 开关在"信息层面"的含义，
而在我们这边它完全由 `cu_seqlens` 决定）：

以实测 preset 的 rank1 为例，本地 token 轴 = 全局 `[512,1024)`，本地
`cu_seqlens = [0, 88, 188, 512]`（3 段：全局 `[512,600)` / `[600,700)` / `[700,1024)`，
注意**下标是相对窗口起点的**）：

| 调用 | 给的 `cu_seqlens` | `Nseq = len-1` | 输出 `hm` | 取哪条链 |
| --- | --- | --- | --- | --- |
| 竞品跨卡那种"只喂末段" | `[188, 512]`（2 个元素） | 1 | `[1, HV, 128, 256]` | `hm[0]`，= 竞品 `cu_seqlens[-2:]` 那条 |
| 整窗一次算（我们的默认用法） | `[0, 88, 188, 512]`（4 个元素） | 3 | `[3, HV, 128, 256]` | `hm[2]`（末段，**与上面那条逐位相同**）；`hm[0]`/`hm[1]` 顺带算出，用来填并行度 |

两点结论：

1. **`MULTI_SEQS` 不是"模式"，而是"这次调用里有几个独立工件"**——每个段一条链（各自的
   `bos/eos`、自己的 `h=0`/`m=I`、自己的 `NT`），所以"一次算 N 段"在数学上就等于"分别算 N 次"，
   这也是 `check_cp_alignment.py` 的 A3 能测到 `max_abs=0.000e+00` 的原因；
2. 我们的算子**只看 `cu_seqlens`**：`Nseq = len(cu_seqlens)-1`，给 2 个元素就产 1 条链
   （≡ 竞品 `MULTI_SEQS=False`），给 N+1 个元素就产 N 条链（≡ 竞品 `MULTI_SEQS=True`）。

**zigzag 到底传什么 `cu_seqlens`**（同一个 preset，`W=2`，`part_len=256`，`T_local=512`）：

```
rank0：本地 buffer = 全局 [0,256) ∪ [768,1024)
       传 ctx.cu_seqlens = [0, 40, 256, 512]        → Nseq=3
       front 段数 fns = 2（front_cu=[0,40,256]）
       front 末段 = hm[fns-1] = hm[1]   （本地[40,256) = 全局[40,256)）
       back  末段 = hm[Nseq-1] = hm[2]  （本地[256,512) = 全局[768,1024)）

rank1：本地 buffer = 全局 [256,768)
       传 ctx.cu_seqlens = [0, 256, 344, 444, 512]  → Nseq=4
       front 段数 fns = 1
       front 末段 = hm[0]  （本地[0,256)  = 全局[256,512)）
       back  末段 = hm[3]  （本地[444,512) = 全局[700,768)）
```

要点：

1. **传的是"本地"数组**（`build_cp_context(...).cu_seqlens`，值域是本地 token 下标 `0..T_local`），
   **不是**建 context 时用的全局 `cu_seqlens_global`；
2. 整窗传参时 `cu_seqlens[0] = 0`、`cu_seqlens[-1] = T_local = 2·part_len` ✓ 满足我们的校验，
   所以**不需要子区间支持**（§5.2 第 12 项只在"逐 part 调"时才需要）；
3. part 边界 `part_len = 256` 就在数组里（rank0 的第 3 个元素、rank1 的第 2 个元素），所以段枚举
   天然按 part 分开，不会跨 part 并段。

### 0.2 `cu_seqlens` 是在哪儿构造的（竞品三级）

**结论：全局边界由调用方给，*本地*边界由 `build_cp_context` 构造，然后调用方把本地边界显式传进算子。**
算子自己不切段、也不做 clamp。

| 级别 | 谁构造 | 内容 |
| --- | --- | --- |
| ① 全局边界 | 调用方（数据打包 / 框架） | 打包轴上每段序列的累计边界 `[0, s1, s1+s2, …, total]`（测试里是 `[0] + cumsum(lengths)`） |
| ② 本地边界 | `fla/ops/cp/context.py::get_cp_cu_seqlens`（经 `build_cp_context`，`context.py:129-207`） | 把全局边界**投影到本 rank 的窗口**：`_interval_cp_meta`（`context.py:87-125`）做 `searchsorted` 找重叠段 → `clamp(start,end)` → 平移到本地坐标 → `unique_consecutive` → `int32`；同时算出 `is_first_rank/is_last_rank`、`pre_num_ranks/post_num_ranks`、`pre_num_conv_tokens`（zigzag 则按 part 各一套，并加 `front_num_seqs`） |
| ③ 传给算子 | 调用方显式传（如 `test_gdn_with_cp.py:719` `cu_seqlens=cp_context.cu_seqlens`） | **本地**累计边界；对应张量也是本地切片 |
| ④ wrapper 内部再切 | `chunk_gated_delta_rule_fwd_h_pre_process` | 用 `cu_seqlens[-2:]` / `cu_seqlens[fns-1:fns+1]` 取出"要导出的那一段"，喂给 kernel |

**zigzag 的本地边界构造公式**（`get_cp_cu_seqlens` 里）：

**"全局 → 本地"的转换机制**（`_interval_cp_meta`，对每个 rank 的窗口 `[start, end)` 做一次）：

```
a. searchsorted 找交集：start_seq_idx = searchsorted(cu[1:],  start, 'right')   # 含窗口起点的段
                        end_seq_idx   = searchsorted(cu[:-1], end,   'left')    # 含窗口终点的段
b. subset = cu[start_seq_idx : end_seq_idx + 1]        # 这些是"落在窗口附近/内部"的全局边界
c. clamp ：subset.clamp(min=start, max=end)            # 窗口外的边界钳到窗口两端
d. shift ：- start                                     # 平移到本地坐标（窗口起点 = 0）
e. unique_consecutive()                                # 去掉 clamp 产生的重复值
f. 转 int32 → 作为 ctx.cu_seqlens（本地）；同时记 pre_num_conv_tokens / is_first / is_last / pre/post_ranks
```

**实测 preset 的完整 trace**（`segs=[0,40,600,700,1024]`, contiguous, `W=2`, `part_len=512`）：

```
rank0  窗口 [0,512)：subset = cu[0:3] = [0,40,600] → clamp(0,512) → [0,40,512] → -0 → [0,40,512]
       本地[0,40)  = 全局[0,40)    （seq0 全段）
       本地[40,512)= 全局[40,512)  （seq1 的头 472 token）

rank1  窗口 [512,1024)：subset = cu[1:5] = [40,600,700,1024] → clamp(512,1024) → [512,600,700,1024]
                       → -512 → [0,88,188,512]
       本地[0,88)  = 全局[512,600)（seq1 的尾巴）  本地[88,188)=全局[600,700)（seq2）
       本地[188,512)=全局[700,1024)（seq3）
```

两个可直接验证的结论：**本地数组恒满足 `local[0]=0`、`local[-1]=T_local`**（clamp+shift 的必然结果），
所以"整窗传参"对竞品本地数组是**零适配**的；而竞品自己那两次调用用的是**子区间**
（`[fns-1:fns+1]` 给的是本地坐标下的相对下标，`bos` 可 > 0、`eos` 可 < `T_local`）。

```python
front_cu = _interval_cp_meta(cu_cpu, part_len * rank,     part_len * (rank + 1),     rank, part_len)[0]
back_cu  = _interval_cp_meta(cu_cpu, part_len * back_pos, part_len * (back_pos + 1), back_pos, part_len)[0]
local    = torch.cat([front_cu, back_cu[1:] + part_len])      # back 段平移 part_len 后接上
# 记录 front_num_seqs = len(front_cu) - 1，供 wrapper 取 "front 末段" 用
```

用实测 preset（`segs=[0,40,600,700,1024]`, `W=2` → `part_len=256`）走一遍 rank1：

```
front 区间 = [256,512)  → front_cu = [0, 256]              （窗口起点落在 seq1 内部 → clip 出 0..256）
back  区间 = [512,768)  → back_cu  = [0, 88, 188, 256]      （seq1 尾 / seq2 / seq3 头）
local = cat([0,256], [88,188,256] + 256) = [0,256,344,444,512]   ✓ 与实测一致
front_num_seqs = 1
```

**与我们的对应**：我们收的是 **host `list[int]`**（对应 `ctx.cu_seqlens_cpu`，见 `api.md` 3.2），
不需要 context 的 `pre/post_num_ranks` 等编排元数据；但 `is_first_rank` / `is_last_rank`
决定"要不要调我们、要不要做 merge"（integration_flow §2）。**切勿把全局边界传给算子**——下标
会越界或取错段。

**contiguous 分支里 `cu_last` 是怎么算出来的（三步）**：

```python
cu_last = cu_seqlens[-2:]                       # ① 默认：本地轴"最后一段"的 [bos, eos)
if use_graph:                                   # ② graph 模式修正：躲开零长 padding 段
    i_last  = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1   # 最后一个"长度>0"的段号
    cu_last = cu_seqlens.index_select(0, torch.stack((i_last, i_last + 1)))
if use_graph or not context.is_last_rank:       # ③ 发 kernel（见 §2：is_last 时没人要它的链）
    ... pre_process_fwd_kernel_merged(... cu_seqlens=cu_last, MULTI_SEQS=False ...)
ag_hm, _ = all_gather_into_tensor(hm, group=context.group)   # ④ 所有 rank 都必须参与
```

要点：

1. **"最后一段"= 本 rank 要交出去的那条链**（跨右边界、继续往右延伸的序列）；其余段在窗口内就
   结束了，没人要。`MULTI_SEQS=False` + 2 个元素 = "这个窗口只含这一段"。
2. **`T` 参数其实被忽略**：varlen 分支里 `T = eos - bos`，真正决定窗口的是 `cu_last`；`T=T` 只是
   传本地张量的 T 维（用于 stride）。所以"窗口 = 张量的子区间"在竞品里天然成立（`bos` 可 > 0）。
3. **graph 模式为什么要重取**：graph 捕获时 host 侧 flag 被冻结，所以改成"总是 launch"（注释：
   *a last rank's hm is never read, but the all-gather needs every rank*）；同时变长 batch 常把
   `cu_seqlens` **padding 到固定长度**，末尾会出现**零长段**（`[x, x]`），此时 `cu_seqlens[-2:]`
   会指向空段 → 所以要按"非空段"重取，并且用 `index_select` 保证**形状恒为 2**（graph 友好）。
   注意这一步是在 **device 张量**上做的（避免 D2H），所以竞品的 `cu_seqlens` 必须是 device tensor。
4. **`is_last_rank` 时也照样 `all_gather`**：它是集合通信，所有 rank 都要参与（`hm` 全 0 也照发）。

## 1. 触发链路

| 入口 | 触发条件 | 传给 pre_process 的实参 |
| --- | --- | --- |
| GDN | `fla/ops/gated_delta_rule/chunk.py:82` `if cp_context is not None:` | `k=k, w=w, u=u, g=g, cu_seqlens, initial_state, context, state_v_first, chunk_size`（`USE_G` 路径，k 是 raw key） |
| KDA | `fla/ops/kda/chunk_fwd.py:86` 同上 | `k=kg, w=w, u=u, gk=g, ...`（`USE_GK` 路径，`kg` 已预 gate） |
| DPLR | `fla/ops/generalized_delta_rule/dplr/chunk.py:100` | `k=kg, w=w, u=u, gk=gi, bg=bg, v=v, ...`（`USE_GK`+`USE_BG`） |
| GDN2 / gated_delta_product | `fla/ops/gdn2/chunk_fwd.py` 同类调用 | 同 GDN 形态 |
| **卡内 CP（推理 prefill）** | `fla/ops/common/intracard_cp.py::intracard_fwd_h` → `intracard_pre_scan`（`MULTI_SEQS=True`） | 把长序列切成 `S_split` 段，一次算 `[S_split,HV,K,V+K]` |

共同前置：`cp_context is None or cp_context.group is None` 时，`cp/chunk_delta_h.py:778` 直接
`return initial_state`，**kernel 不执行**。所以本算子只在 CP 下被触发。

## 2. 每个 rank 的两条判定（决定"要不要算、要不要收"）

`cp/context.py::_interval_cp_meta` 在每个 rank 上算出两个 flag，wrapper 用它们跳过工作：

| flag | 含义 | wrapper 里的动作（`cp/chunk_delta_h.py`） |
| --- | --- | --- |
| `is_last_rank` | 本窗口**末端**的序列不往右延伸（没人要我的链） | `if use_graph or not is_last_rank:` 才发 pre_process kernel（873–915 行） |
| `is_first_rank` | 本窗口**起点**的序列不来自左边（没有前缀要接） | `if use_graph or not is_first_rank:` 才做 merge（916–940 行） |

zigzag 布局下这两组 flag 按 part 给（`is_first_by_part` / `is_last_by_part`），**每个 part 各一次**
kernel/merge（`cp/chunk_delta_h.py:811-846`）。

## 3. 算什么

输入是**一个窗口**（`cu_seqlens=[bos,eos]` 是本窗口在 token 轴上的位置）与窗口里的张量；输出是一对矩阵：

| 输出 | 位置 | 含义 |
| --- | --- | --- |
| `h` | `hm[..., 0:V]` | 假设窗口开头状态为 0，本窗口自己累积出来的状态（`K×V`） |
| `m` | `hm[..., V:V+K]` | 窗口的**传递矩阵**（`K×K`），初值单位阵 `I`，每 chunk `m ← M_c m` |

> **这一层计算本身不含任何 CP 逻辑**——没有通信、没有 part 概念、没有 `is_first/is_last`；
> 它只是"把一段 token 窗口压成一次函数 `(h, m)`"。CP 相关的部分全在包装层（§1、§2、§4），
> 这正是我们能把它单独做成一个算子的原因。反过来，**非 CP 时这个函数根本不会执行**
> （调用点自带 `if cp_context is not None:`，函数内部还有 `if context is None …: return`，
> `cp/chunk_delta_h.py:778-779`）；非 CP 下状态链由下游 `chunk_gated_delta_rule_fwd_h`
> 自己从零状态逐 chunk 递推。

数值契约（`docs/api.md` 第 4 节，必须逐位一致）：
1. `h` 进 Cube（`w @ h`）前降 BF16；2. `v_new` 进 Cube（`kᵀ v_new`）前降 BF16；
3. `m` 的链与 `h` 的累加全程 FP32；4. `hm` 输出 FP32。

## 4. 结果怎么被使用（完整数据流）

```
rank r:  pre_process kernel  →  hm_r = [h_r | m_r]     只含"本窗口末段"那条链
             │
             ├─ all_gather_into_tensor(hm) ──► ag_hm[W, HV, K, V+K]        (cp/comm.py)
             │
             └─ merge_fwd_bwd_kernel(ag_hm, pre_or_post_num_ranks=pre_num_ranks, rank=r)
                    S ← 0
                    for j = r-n_pre … r-1:   S ← M_j · S + he_j            ← README 的复合公式
                    → 写 initial_state[0]（contiguous）/ [0] 与 [fns]（zigzag）
             │
             ▼
        initial_state[N,HV,K,V]      N = 本 rank 的序列段数；只有"跨左边界那条"非零
             │
             ▼
        chunk_gated_delta_rule_fwd_h(..., initial_state=initial_state)         (common/chunk_delta_h.py:727)
             └─ chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](h0=initial_state, ...)
                    USE_INITIAL_STATE=True 时： b_h += tl.load(h0 + ...)      ← 真正被"用"的一步
                    grid = (cdiv(V,BV) * N * HV,)   ← 序列并行度落在下游算子
             │
             ▼
        compress_h0(initial_state, context)   # save_for_backward 省内存：只留非零的 1~2 条
             └─ 反向 expand_h0(...) 还原成 [N,...]                       (cp/chunk_delta_h.py)
```

`compress_h0` 的判据直接印证了"只有一条链有用"：单段窗口（`len(cu_seqlens)==2`）原样返回；
contiguous 多段取 `h0[:1]`；zigzag 取 `h0[[0, front_num_seqs]]`。

同类结构在反向还有一套（`chunk_gated_delta_rule_bwd_dhu_pre_process` + merge 出 `dht`），
**不在本算子范围内**（`docs/api.md` 第 2 节"不做"）。

## 5. 卡内 CP（推理）怎么用同一份输出

`intracard_pre_scan` 一次产出 `[S_split,HV,K,V+K]`，紧接着 `intracard_merge`：

- 只给**非首段**的子序列算初始状态（`h0_seq_ids` / `merge_seq_offsets` / `init_offsets` 是预计算好的索引表）；
- 首段仍然从零状态开始；
- 目的：把长序列的串行 scan 切成 `S_split` 段并行，**关键路径 ÷S**（`compute_subseq_len` 注释原话），
  段数上限 `MAX_SUBSEQS=32`（"Limits merge chain depth to control precision loss"）。

⇒ 与跨卡路径的差异只在"谁跨界"：跨卡是**天然**跨界（长序列被卡切开），卡内是**人为**跨界；
两者对本算子的要求相同——**每段一条链，`[Nseg, HV, K, V+K]`**。

## 6. 触发判据汇总

| 情形 | 需要本算子吗 | 说明 |
| --- | --- | --- |
| 非 CP（`cp_context is None`） | **不需要** | wrapper 直接 return，本算子不执行。定长 `B>1` 的模型场景属于这一类 |
| 有序列跨过 rank 边界（长序列 / 打包不对齐） | **需要** | 内部 rank 既导入又导出；rank0 只导出；最后一个只导入 |
| 每条序列都完整落在某个 rank 内（等长 + `W` 整除 `B`） | **不需要** | 全部 `is_first & is_last`，输出恒 0（实测 `--preset aligned`） |
| 卡内切段（推理 prefill） | **需要**，且全部段都要 | 人为跨界，段数 4~32 |

## 7. 映射到我们的算子（框架侧要做什么）

### 7.1 语义衔接的三段契约（我们 ↔ 编排层 ↔ 下游）

算子之间**没有隐藏约定**：每一段靠一个显式张量/文档接口衔接。

| 段 | 提供方 | 契约内容 |
| --- | --- | --- |
| ① 我们 → 编排层 | 本算子 | `hm[Nseq, HV, K, V+K]` FP32：**每段一条链**，`hm[i]` 的 `[0,V)` 是 `h`、`[V,V+K)` 是 `m`；`Nseq = len(cu_seqlens)-1`，段序 = `cu_seqlens` 的段序 |
| ② 编排层内部 | 框架侧（通信 + merge） | `all_gather(hm)` → 用 `pre_num_ranks`（zigzag 用 `pre_num_ranks_by_part`）按 `S ← M_j·S + he_j` 复合 → 写 `initial_state` |
| ③ 编排层 → 下游 | 框架侧 | `initial_state[N, HV, K, V]` FP32：**每条序列在窗口起点处的状态**；窗口内自己开始的序列填 0，只有"跨左边界那条"非零 |
| ④ 下游 | `chunk_gated_delta_rule_fwd_h` | 收 `initial_state` 当 `h0`（`USE_INITIAL_STATE` 时 `b_h += load(h0 + …)`），配同一套本地 `cu_seqlens` |

所以：

* **下游不需要知道我们的语义细节**——它只要 `initial_state[N,HV,K,V]`，字段语义与竞品完全一致；
* **编排层是唯一的"翻译者"**：它知道 part 结构（`is_first_rank`/`is_last_rank`/`fns`），负责
  "从 `hm` 里挑出每 part 末段那条链 → gather → merge → 填 `initial_state`"；
* **最省事的集成路径**：框架侧照抄竞品 wrapper，但**按"每 part 一个窗口"调我们（`Nseq=1`）**，
  此时 `hm [1,HV,K,V+K]` 去掉 size-1 维就是竞品那块 buffer，后续 all_gather/merge 代码一个字都不用改。
  想要并行度再切到"整窗传参（`Nseq>1`）+ 按索引取两条"。

| 环节 | 竞品在哪做 | 我们 | 结论 |
| --- | --- | --- | --- |
| 建 CP context / 判 first-last | `cp/context.py` | 不做 | 框架侧复刻；并据此决定调不调我们 |
| 单段窗口的 (h, m) | pre_process kernel | **我们的算子** | 形态与竞品**完全一致**：整根张量 + `cu_seqlens=[bos,eos]`（子区间，2026-09-23 定稿），`Nseq=1` 时输出逐字节相同 |
| 整窗多段一次算 | 竞品只用 `MULTI_SEQS`（卡内）；跨卡恒 False | **我们的算子**（前导维 = 链条数） | 第 i 份 == 竞品对第 i 段单独调用（实测逐位相等） |
| all_gather + merge | wrapper | 不做 | 留在框架/通信层 |
| `compress_h0` / `expand_h0` | wrapper | 不做 | 框架侧按需保留/还原 |
| 布局转换（token-major ↔ BNSD） | —— | 调用方吸收 | 我们固定 BNSD |

框架侧调用伪代码：

```python
need_out = not ctx.is_last_rank          # 窗口末端在序列内部 → 要交链出去
need_in  = not ctx.is_first_rank         # 窗口起点在序列内部 → 要接链进来
if need_out:
    hm = op(window=本窗口末段, cu_seqlens=[seg_start, seg_end))   # 或整窗多段一次算
    all_gather(hm)  # 与其它 rank 一起
if need_in:
    merge(...)  → initial_state
```
