"""pre_process_fwd_kernel_merged 的片内容量核算。

固定规格：K = V = 128、BT = 64（host 拦截其它值），一个工作项 = 一个 value head。
只算 GDN / KDA（USE_G / USE_GK）下的 L1 / UB / L0C 占用，与 docs/design.md 3.1 的地址图逐项对应。

换平台时**只改下面 PLATFORM 这一段**（docs/design.md 3.5 的平台常量表）：
  * CAP_* 是硬件容量，取自 docs/precheck.md（platform_config 的 l1_size / ub_size / l0_*_size）；
  * RESERVE_* 是本设计自定的预留，**没有硬件依据**，换硬件必须重新按组件实际占用核对；
  * AIC_NUM 不参与本脚本的容量计算，只在分核时用，host 侧应改为运行时从设备读取。
"""
from __future__ import annotations

KI = 1024
K = V = 128
BT = 64

PLATFORM = {
    "soc": "Ascend950PR_9579",
    "npu_arch": 3510,
    "aic_num": 28,          # host 侧应运行时读取，不写死
    "aiv_num": 56,
    "l1_cap": 512 * KI,
    "ub_cap": 248 * KI,
    "l0a_cap": 64 * KI,
    "l0b_cap": 64 * KI,
    "l0c_cap": 256 * KI,
    "l1_reserve": 64 * KI,  # 本设计自定
    "ub_reserve": 24 * KI,  # 本设计自定
}
L1_BUDGET = PLATFORM["l1_cap"] - PLATFORM["l1_reserve"]
UB_BUDGET = PLATFORM["ub_cap"] - PLATFORM["ub_reserve"]


def l1_usage(mode: str) -> dict[str, int]:
    """L1 固定地址图（3.1.1），单位 KiB。地址与 GATE_MODE 无关，未用的槽保留但不访问。"""
    return {
        "L1[0,64)    m":      K * K * 4 // KI,
        "L1[64,96)   H_c":    K * V * 2 // KI,
        "L1[96,112)  k":      BT * K * 2 // KI,
        "L1[112,128) L_c":    BT * K * 2 // KI,
        "L1[128,144) w":      BT * K * 2 // KI,
        "L1[144,160) v_new":  BT * V * 2 // KI,
        "L1[160,176) bg":     BT * K * 2 // KI,
        "L1[176,192) V_c":    BT * V * 2 // KI,
        "L1[192,256) M_c":    K * K * 4 // KI,
    }


def ub_usage(mode: str) -> dict[str, int]:
    """UB 固定地址图（3.1.2），每个 AIV，单位 KiB。"""
    return {
        "UB[0,32)    h 分片":      K * (V // 2) * 4 // KI,
        "UB[32,64)   dH staging":  K * (V // 2) * 4 // KI,
        "UB[64,96)   Kw staging":  (K // 2) * K * 4 // KI,
        "UB[96,112)  P 分片":      BT * (V // 2) * 4 // KI,
        "UB[112,128) v_new 缓冲":  BT * (V // 2) * 4 // KI,
        "UB[128,132) 小量":        4,
    }


def l0c_peak() -> int:
    """S2 的 dH 与 Kw 的 FP32 累加器同时存活，单位 KiB。"""
    return (K * V * 4 + K * K * 4) // KI


def l1_peak(mode: str) -> int:
    return {"USE_GK": 192, "USE_G": 208, "USE_BG": 256}[mode]


def main() -> None:
    print(f"固定规格 K={K} V={V} BT={BT}")
    for mode in ("USE_GK", "USE_G", "USE_BG"):
        l1_sum, ub_sum = l1_peak(mode), sum(ub_usage(mode).values())
        assert sum(l1_usage(mode).values()) == 256, sum(l1_usage(mode).values())
        assert ub_sum == 132, ub_sum
        assert l1_sum <= L1_BUDGET // KI and ub_sum <= UB_BUDGET // KI
        print(f"[{mode}] L1 峰值 {l1_sum:>3} KiB   UB 峰值(每 AIV) {ub_sum:>3} KiB   "
              f"L0C 阶段内峰值 {l0c_peak():>3} KiB")

    print("\n=== 逐项占用（KiB，USE_BG 最紧档位）===")
    for name, size in {**l1_usage("USE_BG"), **ub_usage("USE_BG")}.items():
        print(f"  {name:<26} {size:>4}")
    p = PLATFORM
    print(f"\n平台 {p['soc']}（NpuArch {p['npu_arch']}，AIC {p['aic_num']} / AIV {p['aiv_num']}）")
    print(f"硬件上限：L1 {p['l1_cap'] // KI} / UB {p['ub_cap'] // KI} / L0A {p['l0a_cap'] // KI} / "
          f"L0B {p['l0b_cap'] // KI} / L0C {p['l0c_cap'] // KI} KiB")
    print(f"设计上限（减预留）：L1 {L1_BUDGET // KI} / UB {UB_BUDGET // KI} KiB")
    print(f"最紧档位 USE_BG 余量：L1 {L1_BUDGET // KI - 256} KiB、"
          f"UB {UB_BUDGET // KI - 132} KiB、L0C {p['l0c_cap'] // KI - l0c_peak()} KiB")


if __name__ == "__main__":
    main()
