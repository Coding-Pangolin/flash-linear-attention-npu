"""pre_process_fwd_kernel_merged 的 msprof op simulator 采集用最小用例。

要求：
  * 只跑一次（无 warmup 循环），shape 取最小可复现规模 —— 仿真逐指令执行，shape 大了会超时。
  * 只能 device 0（仿真不支持多卡/改可见卡号）。
  * `import fla_npu_opp_env` 必须在 `import torch_npu` 之前（把 wheel 内嵌 OPP 写进
    ASCEND_CUSTOM_OPP_PATH，CANN 初始化后设置就来不及了）。

用法（容器内）：
    shape=64  python3 sim_ppfm.py          # 1 段 1 head，1 个 chunk
    shape=128 python3 sim_ppfm.py          # 2 段? 不，T=128/HV=2 → 2 chunk × 2 head
也可显式给： PPFM_SIM_T=64 PPFM_SIM_HV=1 python3 sim_ppfm.py
"""

from __future__ import annotations

import os
import time

# ⚠ 千万不要在这里设 ASCEND_RT_VISIBLE_DEVICES：
#   仿真只支持"0 卡且不修改可见卡号"，显式设置会让 aclInit 报 507000。
#   真机跑的时候可以用环境变量在外部指定。

import fla_npu_opp_env  # noqa: E402,F401  (必须早于 torch_npu)
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def main() -> int:
    T = int(os.environ.get("PPFM_SIM_T", "64"))
    HV = int(os.environ.get("PPFM_SIM_HV", "1"))
    HK = int(os.environ.get("PPFM_SIM_HK", str(HV)))
    K = V = 128
    BT = 64
    variant = os.environ.get("PPFM_SIM_VARIANT", "gdn")   # gdn | kda
    dev = "npu:0"

    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    torch.npu.set_device(0)
    gen = torch.Generator().manual_seed(0)
    k = torch.nn.functional.normalize(
        torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    v = torch.randn(T, HV, V, generator=gen).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).bfloat16()
    nblk = -(-T // BT)
    gate = (-0.013 / BT * (1 + torch.rand(nblk, BT, HV, generator=gen) * 0.5))
    gate = gate.cumsum(1).reshape(-1, HV)[:T].contiguous()

    to_npu = lambda x: x.movedim(1, 0).unsqueeze(0).contiguous().to(dev)
    kd, wd, vd = to_npu(k), to_npu(w), to_npu(v)
    kwargs = dict(cu_seqlens=[0, T], chunk_size=BT)
    if variant == "kda":
        gk = torch.stack([gate] * K, dim=-1).contiguous()      # [T,HV,K] 粗略构造
        kwargs["gk"] = to_npu(gk)
    else:
        kwargs["g"] = to_npu(gate)

    print(f"[sim] shape T={T} HK={HK} HV={HV} K={K} V={V} BT={BT} variant={variant}")
    t0 = time.perf_counter()
    hm = npu_ppfm(kd, wd, vd, **kwargs)          # 单次调用，无 warmup
    torch.npu.synchronize()
    print(f"[sim] hm shape={tuple(hm.shape)} dtype={hm.dtype} "
          f"finite={bool(torch.isfinite(hm).all())} host_elapsed={time.perf_counter() - t0:.2f}s")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
