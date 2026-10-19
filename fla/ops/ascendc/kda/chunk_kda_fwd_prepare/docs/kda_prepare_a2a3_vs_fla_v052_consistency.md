# chunk_kda_fwd_prepare：我们的标杆 vs fla v0.5.2 一致性核对

结论：**一致**。

- **公式级**：逐条对得上（见 §2）；差异只有 3 处，且都不改变数学（见 §4）。
- **数值级**：在同一份输入上，用 fla v0.5.2 的口径重算一遍（脚本
  `cmp_atk_vs_fla.py`），`gk` **完全逐位相同**，`qg/kg/qg_scaled` 实质相同（≤1e-24），
  `aqk/akk/w/u` 差 **≤ 1 个 bf16 ULP**（bf16 的 1 ULP ≈ 3.9e-3 相对）。
  残余差异来自"求逆分块 + 累加精度/顺序"（我们 fp64 标杆里对同一数学用不同分块路径）。

## 1. 对标来源（fla v0.5.2）

| fla 文件 / kernel | 对应我们的 stage |
|---|---|
| `fla/ops/kda/gate.py::kda_gate_fwd_kernel`（gate 激活）、`kda_gate_chunk_cumsum`（`scale=RCP_LN2`） | Stage 0 |
| `fla/ops/kda/chunk.py::l2norm_fwd`（q/k L2norm 在 layer 层做） | Stage 0（我们可选进 kernel） |
| `fla/ops/kda/chunk_intra.py::chunk_kda_fwd_kernel_intra_sub_chunk`（`b_gq/b_gk`、`b_Aqk/b_Akk`、mask） | Stage 1 / 2 / 3 |
| `fla/ops/kda/chunk_intra.py::chunk_kda_fwd_kernel_inter_solve_fused`（16 行前代 + inter-solve） | Stage 3 / 4 / 5 |
| `fla/ops/kda/wy_fast.py::recompute_w_u_fwd_kda_kernel`（`b_qg/b_kg/b_kb/b_vb`、`b_w/b_u`） | Stage 6 / 7 |

## 2. 逐条对照（公式级）

| 项 | 我们（ATK 标杆） | fla v0.5.2 | 判定 |
|---|---|---|---|
| q/k L2norm | `q_hat = bf16(q·rsqrt(Σq²+ε))` | `l2norm_fwd` 同式 | 一致 |
| gate 激活 | `−exp(A_log)·softplus(g+dt_bias)`；`safe_gate` 时 `lower_bound·sigmoid(exp(A_log)·(g+dt_bias))` | `b_yg = -exp(b_A)*softplus(b_g)` / `lower_bound*sigmoid(exp(b_A)*b_g)` | 一致 |
| 不启用 gate | `g_step = g` | `use_gate_in_kernel=False` 直接对 g 做 cumsum | 一致 |
| chunk 内 cumsum | `cumsum(g_step)×(1/ln2 if use_exp2 else 1)` | `tl.cumsum(...)` 后 `*= scale`，调用处 `scale=RCP_LN2` | 一致（我们多支持自然对数域） |
| 子带参考行 | 子带中点 `(begin+end)//2` | 满子带同样取中点（`i_ti + min(BC//2, T-i_ti-1)`） | 满子带一致；**尾块取法不同**（§4-②） |
| 因子 | `fq = E(g−G_ref)`、`fk = E(G_ref−g)` | `b_gq = exp2(b_g−b_gn)`、`b_gk = exp2(-(b_g−b_gn))` | 一致 |
| 打分 | `Aqk_raw = (q_hat⊙fq)@(k_hat⊙fk)ᵀ`、`Akk_raw = (k_hat⊙fq)@(k_hat⊙fk)ᵀ` | `b_Aqk = dot(b_q*b_gq, trans(b_k*b_gk))`、`b_Akk = dot(b_k*b_gq, ...)` | 一致 |
| mask / scale / beta | `Aqk = tril(Aqk_raw·scale, 0)`（i≥j）、`L = tril(Akk_raw·beta, −1)`（i>j） | `m_Aqk = i≥j`、`m_Akk = i>j`，`* scale` / `* b_beta[:,None]` | 一致 |
| 逆矩阵 | `Akk = (I+L)⁻¹`，两个 32×32 叶块 + 跨叶一次乘 | `(I+L)⁻¹`，16×16 对角块前代 + `inter_solve_fused` | 数学一致；**分块/舍入路径不同**（§4-①） |
| qg / kg | `qg = q_hat⊙E(G)`、`kg = k_hat⊙E(G_last−G)` | `b_qg = b_q*exp2(b_gk)`、`b_kg = b_k*exp2(b_gn[None,:]−b_gk)` | 一致 |
| k_beta_g / v_beta | `k_hat⊙E(G)⊙beta`、`v⊙beta` | `b_kb = (k*beta)*exp2(b_gk)`、`b_vb = v*beta` | 一致（乘的顺序不同，同一个积） |
| w / u | `w = Akk@k_beta_g`、`u = Akk@v_beta` | `b_w = dot(b_A, b_kb)`、`b_u = dot(b_A, b_vb)` | 一致 |
| 输出集合 | gk/aqk/akk/w/u/qg/kg/qg_scaled/q_hat/k_hat/q_rstd/k_rstd/beta_eff | 只出 w/u/qg/kg/Aqk/Akk（其余在 layer/backward 里算） | 我们多出反向用中间量（§4-③） |

## 3. 数值核对（同一份输入，fla 口径 vs 我们的标杆）

脚本：`python3 cmp_atk_vs_fla.py <case_json> <cid> <cache.pt>`（fp64 重算，bf16 落点与 fla 一致）。

| case | 覆盖 | gk | aqk | akk | w | u | qg | kg | qg_scaled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 模型规模、l2norm、exp2 | **0** | 4.883e-04 (5.5e-3) | — | 1.221e-04 (5.1e-3) | 4.883e-04 (4.1e-3) | — | 1.5e-25 | 1.3e-26 |
| 61 | 模型规模、17% 因子越界 | **0** | 4.883e-04 (5.5e-3) | — | 2.441e-04 (5.2e-3) | 9.766e-04 (3.0e-3) | — | 3.6e-25 | 3.2e-26 |
| 21 | 自然对数域 gate | **0** | 7.812e-03 (5.7e-3) | 9.766e-04 | 4.883e-04 | 4.883e-04 | **0** | **0** | **0** |
| 23 | safe_gate + exp2 | **0** | 3.125e-02 (2.8e-3) | 4.883e-04 | 4.883e-04 | 2.441e-04 | 7.6e-25 | 8.5e-25 | 7.6e-25 |
| 24 | beta sigmoid | **0** | 9.766e-04 (3.0e-3) | 2.441e-04 | 4.883e-04 | 2.441e-04 | **0** | **0** | **0** |
| 25 | beta 2×sigmoid | **0** | 7.812e-03 | 1.953e-03 | 1.953e-03 | 9.766e-04 | **0** | **0** | **0** |
| 68 | l2norm + 2×sigmoid + GVA(h2 hv4) | **0** | 9.766e-04 (9.8e-4) | — | 1.953e-03 | 1.953e-03 | — | **0** | **0** |

（括号内为相对误差；"—"表示该用例的 `backward_mode` 不返回这一项；1 个 bf16 ULP ≈ 3.9e-3。）

## 4. 三处差异，逐条给结论

① **求逆分块**：我们按两个 32×32 叶块 `X0/X1` + 跨叶 `−X1@(L[32:M,0:32]@X0)`；
fla 按 16×16 对角块做前代、再用 `inter_solve` 合并。两者都是 `(I+L)⁻¹`，
差别只在浮点分块/舍入路径 ⇒ 落在 1 ULP 内（§3 的 aqk/akk/w/u）。

② **尾块（部分子带）参考行**：fla 取 `i_ti + min(BC//2, T−i_ti−1)`（不足 16 行时取该子带最后一个有效行），
我们取 `(begin+end)//2`（有效行的中点）。参考行只影响数值条件（避免 exp2 溢出），
数学上与结果无关——case 61 里存在部分子带，实测仍 ≤1 ULP（§3）⇒ **不影响一致性**。

③ **指数 clamp**：我们的标杆把指数钳在 `[−126,120]`（qg/kg 侧 `[−80,80]`），fla 不钳。
实测越界只发生在"gate 单调递减"造成的**极小值方向**（case 61 有 17% 的前缀因子 `|G_ref−g|>126`），
钳到 `2^-126` 与 fla 的下溢到 0 之间差 ~1e-38，被 bf16 输出舍入吃掉 ⇒
**对结果无影响**（§3 的 case 61 仍是 1 ULP）。

另外我们比 fla 多出的东西（不是差异，是超集）：`allow_neg_eigval` 的 `2×sigmoid(beta)` 模式、
非 exp2 的自然对数域、以及反向所需的 `q_hat/k_hat/q_rstd/k_rstd/beta_eff/qg_scaled` 输出。

## 5. 复现

```bash
cd /home/admin123/kda_a2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 cmp_atk_vs_fla.py repo/tests/atk/chunk_kda_fwd_prepare/atk_chunk_kda_fwd_prepare_a2.json \
        61 cache/acc61.pt
```
