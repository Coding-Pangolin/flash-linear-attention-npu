"""单 chunk、单 head、g=0 的最小定位用例。

此时数学上应有：
    h = K_c^T @ v_new = K^T @ v     （单 chunk、初始状态为 0）
    m = I - K^T @ W_c
把 NPU 结果分别与「闭式解」和「标杆」对比，判断是 S0/S1（v_new）还是
S2/S3/S4（h/m 链）出错。
"""

from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    op_dir = os.environ.get(
        "PPFM_OP_DIR",
        "/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/"
        "gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged")
    sys.path.insert(0, os.path.join(op_dir, "reference"))
    from reference import pre_process_fwd_kernel_merged as ref_ppfm
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    torch.manual_seed(0)
    T, HK, HV, K, V = 64, 1, 1, 128, 128
    dev = "npu:0"
    k = torch.randn(T, HK, K).to(torch.bfloat16)
    v = torch.randn(T, HV, V).to(torch.bfloat16)
    w = torch.randn(T, HV, K).to(torch.bfloat16)

    ref = ref_ppfm(k, v, w, g=torch.zeros(T, HV), chunk_size=64, cu_seqlens=[0, T])
    ref = ref[0]                      # [K, V+K]
    h_ref, m_ref = ref[:, :V], ref[:, V:]

    # 闭式解（单 chunk、g = 0、初始 h = 0）
    kf, wf, vf = k[:, 0].float(), w[:, 0].float(), v[:, 0].float()
    v_new = vf - (wf @ torch.zeros(K, V))
    h_closed = kf.t() @ v_new
    m_closed = torch.eye(K) - (kf.t() @ wf)

    got = npu_ppfm(
        k.movedim(1, 0).unsqueeze(0).to(dev),
        w.movedim(1, 0).unsqueeze(0).to(dev),
        v.movedim(1, 0).unsqueeze(0).to(dev),
        g=torch.zeros(1, HV, T, dtype=torch.float32, device=dev),
        cu_seqlens=[0, T], chunk_size=64,
    ).cpu()[0, 0]
    torch.npu.synchronize()

    def stat(tag, a, b):
        d = (a - b).abs()
        print(f"{tag:28s} max_abs={d.max().item():.3e}  mean_abs={d.mean().item():.3e}  "
              f"ref_absmax={b.abs().max().item():.3e}")

    print("=== h 半边 ===")
    stat("npu vs 标杆", got[:, :V], h_ref)
    stat("npu vs 闭式 K^T v", got[:, :V], h_closed)
    stat("标杆 vs 闭式", h_ref, h_closed)
    print("=== m 半边 ===")
    stat("npu vs 标杆", got[:, V:], m_ref)
    stat("npu vs 闭式 I-K^T W", got[:, V:], m_closed)
    stat("标杆 vs 闭式", m_ref, m_closed)

    print("h npu 前 3 行前 4 列:", got[:3, :4].tolist())
    print("h ref 前 3 行前 4 列:", h_ref[:3, :4].tolist())
    print("m npu 对角前 4:", [got[i, V + i].item() for i in range(4)])
    print("m ref 对角前 4:", [m_ref[i, i].item() for i in range(4)])
    m_npu = got[:, V:]
    print("m npu 行0 前6:", [round(x, 4) for x in m_npu[0, :6].tolist()])
    print("m ref 行0 前6:", [round(x, 4) for x in m_ref[0, :6].tolist()])
    print("m npu 行0 尾4:", [round(x, 4) for x in m_npu[0, -4:].tolist()])
    print("m ref 行0 尾4:", [round(x, 4) for x in m_ref[0, -4:].tolist()])
    print("m npu 行1 前6:", [round(x, 4) for x in m_npu[1, :6].tolist()])
    print("m ref 行1 前6:", [round(x, 4) for x in m_ref[1, :6].tolist()])
    bad = (m_npu - m_ref).abs() > 1e-3
    print("m npu 行0 第 88..99 列:", [round(x, 4) for x in m_npu[0, 88:100].tolist()])
    print("m ref 行0 第 88..99 列:", [round(x, 4) for x in m_ref[0, 88:100].tolist()])
    print("m npu 列 93 前 6 行:", [round(x, 4) for x in m_npu[:6, 93].tolist()])
    print("m ref 列 93 前 6 行:", [round(x, 4) for x in m_ref[:6, 93].tolist()])
    print("m npu 列 92 前 6 行:", [round(x, 4) for x in m_npu[:6, 92].tolist()])
    print("m npu 行0 全零列数:", int((m_npu[0].abs() < 1e-12).sum().item()))
    print("h npu 行0 尾4:", [round(x, 4) for x in got[0, -4:].tolist()])
    print("h ref 行0 尾4:", [round(x, 4) for x in h_ref[0, -4:].tolist()])
    print("m 错位元素数:", int(bad.sum().item()), "/", bad.numel())
    rows_bad = bad.any(dim=1).nonzero().flatten().tolist()
    print("出错行号前 10 个:", rows_bad[:10], " 共", len(rows_bad), "行")
    cols_bad = bad.any(dim=0).nonzero().flatten().tolist()
    print("出错列号前 10 个:", cols_bad[:10], " 共", len(cols_bad), "列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
