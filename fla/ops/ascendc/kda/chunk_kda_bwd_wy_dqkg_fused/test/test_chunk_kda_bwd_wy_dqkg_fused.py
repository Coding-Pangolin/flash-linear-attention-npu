# Copyright (c) 2026 Tianjin University, Ltd.
"""CPU golden for ChunkKdaBwdWyDqkgFused (Cube-faithful, exp2)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _exp2(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(x)


def chunk_kda_bwd_wy_dqkg_fused_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BNSD inputs. Returns dq, dk, dv2, dg, db, dA (fp32 grads where Triton uses float)."""
    assert q.dim() == 4
    B, H, T, K = q.shape
    HV, V = v.shape[1], v.shape[-1]
    BT = chunk_size
    dtype = q.dtype
    G = HV // H

    dq = torch.zeros(B, HV, T, K, dtype=torch.float32, device=q.device)
    dk = torch.zeros(B, HV, T, K, dtype=torch.float32, device=q.device)
    dv2 = torch.zeros_like(v)
    dg = torch.zeros(B, HV, T, K, dtype=torch.float32, device=q.device)
    db = torch.zeros(B, HV, T, dtype=torch.float32, device=q.device)
    dA = torch.zeros(B, HV, T, BT, dtype=torch.float32, device=q.device)

    if cu_seqlens is None:
        tasks = [(b, 0, b * T, T, it) for b in range(B) for it in range((T + BT - 1) // BT)]
    else:
        cu = cu_seqlens.tolist()
        idx = chunk_indices.view(-1, 2).tolist()
        tasks = []
        for seq_id, local_chunk in idx:
            bos, eos = cu[seq_id], cu[seq_id + 1]
            tasks.append((0, seq_id, bos, eos - bos, local_chunk))

    for b_idx, _seq, bos, local_t, i_t in tasks:
        t0 = bos + i_t * BT
        valid = min(BT, local_t - i_t * BT)
        if valid <= 0:
            continue
        o_t = torch.arange(valid, device=q.device)
        m_t = o_t < valid
        m_last = o_t == (valid - 1)

        for i_hv in range(HV):
            i_h = i_hv // G
            beta_c = beta[b_idx, i_hv, t0 : t0 + valid].float()
            # A tile [BT,BT]: storage [T,BT] -> row-major left-multiply layout
            A_tile = a[b_idx, i_hv, t0 : t0 + valid, :valid].to(dtype)
            A_full = torch.zeros(BT, BT, dtype=dtype, device=q.device)
            A_full[:valid, :valid] = A_tile

            b_dA = torch.zeros(BT, BT, dtype=torch.float32, device=q.device)
            b_db = torch.zeros(BT, dtype=torch.float32, device=q.device)

            # V0
            dv_c = dv[b_idx, i_hv, t0 : t0 + valid]
            v_c = v[b_idx, i_hv, t0 : t0 + valid]
            b_dA[:valid, :valid] += (dv_c.float() @ v_c.float().T)
            dvb = (A_full[:valid, :valid].float() @ dv_c.float())
            dv2[b_idx, i_hv, t0 : t0 + valid] = (dvb * beta_c[:, None]).to(dtype)
            b_db[:valid] += (dvb * v_c.float()).sum(-1)

            # state for this chunk
            if state_v_first:
                h_c = h[b_idx, i_hv, i_t]  # [V,K]
                dh_c = dh[b_idx, i_hv, i_t]
            else:
                h_c = h[b_idx, i_hv, i_t]  # [K,V]
                dh_c = dh[b_idx, i_hv, i_t]

            BK = 64
            for i_k in range((K + BK - 1) // BK):
                k0 = i_k * BK
                bk = min(BK, K - k0)
                k_tile = k[b_idx, i_h, t0 : t0 + valid, k0 : k0 + bk]
                g_tile = g[b_idx, i_hv, t0 : t0 + valid, k0 : k0 + bk].float()
                gn = g[b_idx, i_hv, t0 + valid - 1, k0 : k0 + bk].float()

                b_dq = torch.zeros(valid, bk, dtype=torch.float32, device=q.device)
                b_dk = torch.zeros(valid, bk, dtype=torch.float32, device=q.device)
                b_dw = torch.zeros(valid, bk, dtype=torch.float32, device=q.device)
                b_dgk = torch.zeros(bk, dtype=torch.float32, device=q.device)

                BV = 128
                for i_v in range((V + BV - 1) // BV):
                    v0 = i_v * BV
                    bv = min(BV, V - v0)
                    do_t = do[b_idx, i_hv, t0 : t0 + valid, v0 : v0 + bv]
                    vn_t = v_new[b_idx, i_hv, t0 : t0 + valid, v0 : v0 + bv]
                    dv_t = dv[b_idx, i_hv, t0 : t0 + valid, v0 : v0 + bv]
                    if state_v_first:
                        h_t = h_c[v0 : v0 + bv, k0 : k0 + bk]
                        dh_t = dh_c[v0 : v0 + bv, k0 : k0 + bk]
                    else:
                        h_t = h_c[k0 : k0 + bk, v0 : v0 + bv].T  # [bv,bk]
                        dh_t = dh_c[k0 : k0 + bk, v0 : v0 + bv].T

                    b_dgk += (h_t.float() * dh_t.float()).sum(0)
                    b_dq += do_t.float() @ h_t.to(do_t.dtype).float()
                    b_dk += vn_t.float() @ dh_t.to(vn_t.dtype).float()
                    b_dw += dv_t.float() @ h_t.to(dv_t.dtype).float()

                gk = _exp2(g_tile)
                b_dgk = b_dgk * _exp2(gn)
                b_dq = b_dq * gk * scale
                mask = m_t[:valid, None]
                b_dk = b_dk * torch.where(mask, _exp2(gn[None, :] - g_tile), torch.zeros_like(b_dk))
                kg = (k_tile.float() * gk).to(dtype)
                b_dw = (-b_dw).to(dtype)

                b_dA[:valid, :valid] += (b_dw.float() @ kg.float().T)
                dkgb = A_full[:valid, :valid].float() @ b_dw.float()
                b_db[:valid] += (dkgb * kg.float()).sum(-1)

                q_tile = q[b_idx, i_h, t0 : t0 + valid, k0 : k0 + bk].float()
                kdk = k_tile.float() * b_dk
                b_dgk = b_dgk + kdk.sum(0)
                last = m_last[:valid, None].float()
                b_dg = q_tile * b_dq - kdk + last * b_dgk + kg.float() * dkgb * beta_c[:, None]
                b_dk = b_dk + dkgb * (gk * beta_c[:, None])

                dq[b_idx, i_hv, t0 : t0 + valid, k0 : k0 + bk] = b_dq
                dk[b_idx, i_hv, t0 : t0 + valid, k0 : k0 + bk] = b_dk
                dg[b_idx, i_hv, t0 : t0 + valid, k0 : k0 + bk] = b_dg

            # dA finalize
            ii = o_t[:, None]
            jj = o_t[None, :]
            m_A = (ii > jj) & (ii < valid) & (jj < valid)
            beta_j = beta_c[None, :].expand(valid, valid)
            tmp = torch.zeros(BT, BT, dtype=torch.float32, device=q.device)
            tmp[:valid, :valid] = torch.where(m_A, b_dA[:valid, :valid] * beta_j, 0.0)
            tmp2 = tmp[:valid, :valid].to(dtype).float() @ A_full[:valid, :valid].float()
            tmp3 = A_full[:valid, :valid].float() @ tmp2.to(dtype).float()
            out = torch.where(m_A, -tmp3, torch.zeros_like(tmp3))
            dA[b_idx, i_hv, t0 : t0 + valid, :valid] = out
            db[b_idx, i_hv, t0 : t0 + valid] = b_db[:valid]

    return dq, dk, dv2, dg, db, dA


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, HV, K, V, BT = 1, 128, 2, 2, 128, 128, 64
    q = torch.randn(B, H, T, K, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, K, dtype=torch.bfloat16)
    v = torch.randn(B, HV, T, V, dtype=torch.bfloat16)
    v_new = torch.randn(B, HV, T, V, dtype=torch.bfloat16)
    g = torch.randn(B, HV, T, K, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(B, HV, T, dtype=torch.bfloat16)
    a = torch.randn(B, HV, T, BT, dtype=torch.bfloat16)
    NT = (T + BT - 1) // BT
    h = torch.randn(B, HV, NT, K, V, dtype=torch.bfloat16)
    dh = torch.randn(B, HV, NT, K, V, dtype=torch.bfloat16)
    do = torch.randn(B, HV, T, V, dtype=torch.bfloat16)
    dv = torch.randn(B, HV, T, V, dtype=torch.bfloat16)
    outs = chunk_kda_bwd_wy_dqkg_fused_ref(
        q, k, v, v_new, g, beta, a, h, dh, do, dv, scale=1.0 / math.sqrt(K), chunk_size=BT
    )
    print([o.shape for o in outs], "ok")
