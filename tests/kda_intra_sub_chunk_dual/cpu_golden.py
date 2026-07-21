"""fp64 CPU golden for chunk_kda_fwd_intra_sub_chunk (BNSD layout, MHA only).

Copied from the AscendC operator test and made importable.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

BC = 16


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq in range(int(cu_seqlens.numel()) - 1):
        length = int(cu_seqlens[seq + 1] - cu_seqlens[seq])
        n_chunks = (length + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            indices.extend([seq, local])
    return indices


def chunk_kda_fwd_intra_sub_chunk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference matching Triton safe-gate diagonal path. Inputs are BNSD [B,H,T,K]."""
    assert q.ndim == 4 and q.shape == k.shape == g.shape
    B, H, T, K = q.shape
    assert beta.shape == (B, H, T)
    assert chunk_size in (32, 64, 128)
    BT = chunk_size
    NC = BT // BC

    q64 = q.double()
    k64 = k.double()
    g64 = g.double()
    beta64 = beta.double()

    aqk = torch.zeros(B, H, T, BT, dtype=torch.float64, device=q.device)
    akkd = torch.zeros(B, H, T, BC, dtype=torch.float64, device=q.device)

    if cu_seqlens is None:
        tasks = []
        nt = (T + BT - 1) // BT
        for i_b in range(B):
            for i_h in range(H):
                for i_t in range(nt):
                    for i_i in range(NC):
                        tasks.append((i_b, i_h, 0, i_t, i_i, i_b * T, T))
    else:
        assert B == 1
        if chunk_indices is None:
            flat = prepare_chunk_indices(cu_seqlens, BT)
            chunk_indices = torch.tensor(flat, dtype=torch.long, device=q.device)
        assert chunk_indices.numel() % 2 == 0
        tasks = []
        for nt_i in range(chunk_indices.numel() // 2):
            seq = int(chunk_indices[nt_i * 2])
            local = int(chunk_indices[nt_i * 2 + 1])
            bos = int(cu_seqlens[seq])
            eos = int(cu_seqlens[seq + 1])
            local_t = eos - bos
            for i_h in range(H):
                for i_i in range(NC):
                    tasks.append((0, i_h, seq, local, i_i, bos, local_t))

    for i_b, i_h, _seq, local_chunk, i_i, bos, local_t in tasks:
        i_ti = local_chunk * BT + i_i * BC
        if i_ti >= local_t:
            continue
        valid = min(BC, local_t - i_ti)
        mid = i_ti + min(BC // 2, local_t - i_ti - 1)

        rows = []
        for r in range(valid):
            tok = bos + i_ti + r
            rows.append(tok)

        g_block = g64[i_b, i_h, rows, :]
        g_mid = g64[i_b, i_h, bos + mid, :]
        gm = g_block - g_mid
        gq = torch.exp2(gm)
        gk = torch.exp2(-gm)
        q_block = q64[i_b, i_h, rows, :] * gq
        k_pos = k64[i_b, i_h, rows, :] * gq
        k_neg = k64[i_b, i_h, rows, :] * gk
        beta_row = beta64[i_b, i_h, rows]

        aqk_blk = (q_block @ k_neg.transpose(0, 1)) * scale
        akk_blk = (k_pos @ k_neg.transpose(0, 1)) * beta_row[:, None]

        tril = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device))
        strict = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device), diagonal=-1)
        aqk_blk = torch.where(tril, aqk_blk, torch.zeros_like(aqk_blk))
        L = torch.where(strict, akk_blk, torch.zeros_like(akk_blk))

        # Forward substitution -> (I-L)^{-1}
        M = -L.clone()
        for i in range(2, valid):
            a = M[i].clone()
            a[i:] = 0
            a = a + a @ M
            a[i:] = 0
            M[i] = a
        M = M + torch.eye(valid, dtype=torch.float64, device=q.device)

        for r, tok in enumerate(rows):
            aqk[i_b, i_h, tok, i_i * BC: i_i * BC + valid] = aqk_blk[r, :valid]
            akkd[i_b, i_h, tok, :valid] = M[r, :valid]
            if valid < BC:
                akkd[i_b, i_h, tok, valid:] = 0

    return aqk, akkd
