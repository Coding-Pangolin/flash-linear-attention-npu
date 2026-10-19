# A2/A3 chunk_kda_fwd_prepare Stage 划分

硬件预算：L1 512 KiB，L0C 128 KiB，每个 AIV 的 UB 192 KiB。

变量名与公式对标 fla v0.5.2：`fla/ops/kda/chunk.py`（l2norm、beta、g_input→g_cumsum）、
`fla/ops/kda/gate.py`（gate 激活 `b_yg` + chunk 内 cumsum，`scale=RCP_LN2`）、
`fla/ops/kda/chunk_intra.py`（`b_gn/b_gm/b_gq/b_gk/b_Aqk/b_Akk`、逆矩阵 `A`）、
`fla/ops/kda/wy_fast.py`（`b_qg/b_kg/b_kb/b_vb/b_w/b_u`）。
`g_cumsum` 在内核里叫 `gk`。

我们的融合：① gate 激活 + chunk 内 cumsum + intra 打分/求逆 + w,u 合成一个算子；
② q/k 的 L2norm 可选进 kernel（对标在 layer 层做）；③ 逆矩阵按两个 32×32 叶块 + 跨叶一次乘
（对标是 16 行前代 + inter-solve）；④ 额外产出反向所需的中间量与 `qg_scaled`。

BT = 64，BC = 16，M 为本 chunk 的有效行数；非 exp2 模式下 exp2 换成 exp（`g_cumsum` 存自然对数域）。

## Stage 0：Vector

```text
# q/k L2norm（对标 l2norm_fwd）
q_rstd = rsqrt(Σd q ⊙ q + ε)
k_rstd = rsqrt(Σd k ⊙ k + ε)
q = q ⊙ q_rstd
k = k ⊙ k_rstd

# gate 激活（对标 kda_gate_fwd_kernel 的 b_yg）
A       = A_log
g_input = g + dt_bias

b_yg = −exp(A) ⊙ softplus(g_input)                    # 默认
     = lower_bound ⊙ sigmoid(exp(A) ⊙ g_input)        # 开了 lower_bound
     = g                                              # 不启用 gate

g_cumsum = cumsum(b_yg) ⊙ (1/ln2)                     # chunk 内累积（内核参数名 gk）

# beta（对标 chunk.py）
beta = beta_raw
     = sigmoid(beta_raw)                              # use_beta_sigmoid_in_kernel
     = 2 ⊙ sigmoid(beta_raw)                          # 且 allow_neg_eigval
```

出参：gk = g_cumsum、q_hat = q、k_hat = k、q_rstd、k_rstd、beta_eff = beta

## Stage 1：Vector

```text
对每个 BC = 16 行子带：

    b_g  = g_cumsum[子带各行]
    b_gn = g_cumsum[i_ti + min(BC//2, T − i_ti − 1)]   # 子带参考行
    b_gm = b_g − b_gn
    b_gq = exp2(b_gm)                                  # 因子（子带行）
    b_gk = exp2(−b_gm)                                 # 因子（前缀行 16/32/48/64）
```

## Stage 2：Cube

```text
b_kgt = (b_k ⊙ b_gk)ᵀ
b_Aqk = (b_q ⊙ b_gq) @ b_kgt
b_Akk = (b_k ⊙ b_gq) @ b_kgt
```

## Stage 3：Vector

```text
Aqk = tril(b_Aqk ⊙ scale, 0)                          # m_Aqk：i ≥ j（含对角）
Akk = tril(b_Akk ⊙ beta, −1)                          # m_Akk：i > j（严格下三角）

# 逆矩阵（对标：Akk 先存 L、再原地换成 A；对标用 16 行前代 + inter-solve）
A = (I + Akk)⁻¹，按两个 32×32 叶块 + 跨叶一次乘：

    X0 = (I + Akk[0:32, 0:32])⁻¹
    X1 = (I + Akk[32:64, 32:64])⁻¹
    A[0:32, 0:32]   = X0
    A[32:64, 32:64] = X1
    A[32:M, 0:32]   = −X1 @ (Akk[32:M, 0:32] @ X0)
```

出参：aqk = Aqk、akk = A

## Stage 4：Cube

```text
b_T = Akk[32:M, 0:32] @ X0                            # 仅 M > 32
```

## Stage 5：Cube

```text
A[32:M, 0:32] = −X1 @ b_T                             # 仅 M > 32
```

## Stage 6：Vector

```text
b_gn = g_cumsum[最后一行]

b_qg = q ⊙ exp2(g_cumsum)
b_kg = k ⊙ exp2(b_gn[None, :] − g_cumsum)
b_kb = (k ⊙ beta) ⊙ exp2(g_cumsum)
b_vb = v ⊙ beta
```

出参：qg = b_qg、kg = b_kg、qg_scaled = b_qg ⊙ scale（我们额外给反向用）

## Stage 7：Cube

```text
b_w = A @ b_kb
b_u = A @ b_vb
```

出参：w = b_w、u = b_u
