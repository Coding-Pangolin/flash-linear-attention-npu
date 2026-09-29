"""msprof op 采集 harness：单次调用（msopprof 自己 replay），shape 由环境变量控制。

用法（容器内，已安装本算子 wheel）：
    PPFM_T=11264 PPFM_HK=32 PPFM_HV=32 PPFM_VARIANT=gdn \
    msprof op --application="python3 <this file>" \
        --output=<out> --kernel-name=PreProcessFwdKernelMerged \
        --launch-count=1 --warm-up=1 --aic-metrics=Default

配套 `run_msopprof.sh` 会按 shape 循环调用本文件并打印 `OpBasicInfo.csv` 的 Task Duration。
注意：**不要**在容器里设 ASCEND_RT_VISIBLE_DEVICES 之外的额外绑定；真卡按容器可见卡跑。
"""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm


def main() -> None:
    t_len = int(os.environ.get("PPFM_T", "11264"))
    hk = int(os.environ.get("PPFM_HK", "32"))
    hv = int(os.environ.get("PPFM_HV", "32"))
    k_dim = int(os.environ.get("PPFM_K", "128"))
    v_dim = int(os.environ.get("PPFM_V", "128"))
    bt = int(os.environ.get("PPFM_BT", "64"))
    variant = os.environ.get("PPFM_VARIANT", "gdn")
    dev = os.environ.get("PPFM_DEV", "npu:0")

    gen = torch.Generator().manual_seed(0)
    k = torch.nn.functional.normalize(
        torch.randn(t_len, hk, k_dim, generator=gen), dim=-1).bfloat16()
    v = torch.randn(t_len, hv, v_dim, generator=gen).bfloat16()
    beta = torch.rand(t_len, hv, 1, generator=gen) * 0.02
    head_of_k = torch.arange(hv) // (hv // hk)
    w = (beta * k[:, head_of_k].float()).bfloat16()
    nblk = -(-t_len // bt)
    gate = (-0.013 / bt * (1 + torch.rand(nblk, bt, hv, generator=gen) * 0.5))
    gate = gate.cumsum(1).reshape(-1, hv)[:t_len].contiguous()

    kd = k.movedim(1, 0).unsqueeze(0).to(dev)
    wd = w.movedim(1, 0).unsqueeze(0).to(dev)
    vd = v.movedim(1, 0).unsqueeze(0).to(dev)

    if variant == "gdn":
        gd = gate.movedim(1, 0).unsqueeze(0).to(dev).float()
        out = npu_ppfm(kd, wd, vd, g=gd, cu_seqlens=[0, t_len], chunk_size=bt)
    else:
        gk = gate.unsqueeze(-1).expand(t_len, hv, k_dim).contiguous()
        gkd = gk.movedim(1, 0).unsqueeze(0).to(dev).float()
        out = npu_ppfm(kd, wd, vd, gk=gkd, cu_seqlens=[0, t_len], chunk_size=bt)
    torch.npu.synchronize()
    print(f"[perf] T={t_len} HK={hk} HV={hv} variant={variant} out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
