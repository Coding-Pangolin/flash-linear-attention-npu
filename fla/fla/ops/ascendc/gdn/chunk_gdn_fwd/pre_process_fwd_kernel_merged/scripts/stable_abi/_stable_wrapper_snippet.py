# ---------------------------------------------------------------------------
# 追加到 fla_npu/ops/ascendc/_stable.py 末尾（apply_stable_abi.py 自动做）
# ---------------------------------------------------------------------------


def npu_pre_process_fwd_kernel_merged(k, w, u, g=None, *, gk=None, bg=None,
                                      v=None, cu_seqlens=None,
                                      chunk_size=None):
    """CP 前处理：把 token 窗口压成仿射链 (h | m)。

    与 aclnn 头文件逐参对齐（stream 固定在最后）：
    k/w/u 必给；g 与 gk 二选一；bg 仅 DPLR 且必须配 gk；v 仅 DPLR 用，
    GDN/KDA 传 None 表示复用 u；cu_seqlens 必给（varlen 打包窗口，
    也允许子区间 0 <= cu[0] < cu[-1] <= T）。返回 hm[Nseq, HV, K, V+K] FP32。
    """

    if cu_seqlens is None:
        raise ValueError(
            "pre_process_fwd_kernel_merged requires cu_seqlens (varlen only): "
            "pass a host int list such as [bos, eos] or [0, s1, s2, ..., T]."
        )
    if (g is None) == (gk is None):
        raise ValueError("exactly one of g / gk must be provided.")
    if bg is not None:
        # TilingKey 3（USE_BG）只注册未实现：传进来会静默按 GDN/KDA 语义算，宁可明确拒绝。
        raise NotImplementedError(
            "DPLR (bg) is reserved and not implemented in this release."
        )
    if chunk_size not in (None, 64):
        raise ValueError(
            f"pre_process_fwd_kernel_merged only supports chunk_size=64, got {chunk_size}."
        )

    return _op("npu_pre_process_fwd_kernel_merged")(
        k, w, u, g, gk, bg, v,
        _host_ints(cu_seqlens),
        64 if chunk_size is None else int(chunk_size),
        _current_stream_ptr(),
    )


# ---------------------------------------------------------------------------
# fla_npu/ops/ascendc/__init__.py：_ASCENDC_OPS 里加一行 public 名
# ---------------------------------------------------------------------------
#     "npu_pre_process_fwd_kernel_merged",
#
# 说明：本算子不在原地写参数（hm 是新建输出），所以不需要登记
# MUTATED_ARGUMENTS / MUTATION_FLAGS；ctypes 回退（_aclnn_ctypes.py 里的
# npu_pre_process_fwd_kernel_merged）保留，_LAUNCHER_ONLY_OPS 会自动推导。
#
# ⚠ 建议同步在 ctypes wrapper（_aclnn_ctypes.py）里加同样的 bg / chunk_size 守卫：
#   现在那边允许 bg 透传，而 kernel 的 TilingKey 3 分支会按 GDN/KDA 语义算，
#   结果是"静默算错"而不是报错。
