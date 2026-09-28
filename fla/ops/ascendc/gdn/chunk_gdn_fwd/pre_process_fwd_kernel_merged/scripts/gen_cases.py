"""为 pre_process_fwd_kernel_merged 生成泛化用例表（精度 + 性能 + 边界）。

固定规格（2026-09-21 定稿）：K = V = 128、BT = 64、layout = BNSD。

序列的两种表达（与仓内 chunk_fwd_h、上游 chunk_delta_h 一致）：
  * 定长（dense）：`cu_seqlens` 缺省，序列数 `Nseq = B`
  * 变长（varlen）：`B = 1`，T 轴打包多段，`cu_seqlens` 给边界，`Nseq = len(cu_seqlens)-1`
并行度 = `Nseq × HV`（chunk 维被依赖砍掉，见 docs/design.md 2.1.4）：`Nwork = Nseq × HV`。
只有 `Nseq = 1` 且 `HV` 小时才会用不满核；吃满多序列是 2.1.4 推荐的首选解法。
用例表里的 `(B, HV)` 配对取自 `GDN泛化用例表.xlsx` 的真实场景。
GVA：HK 与 HV 可以不一致但必须成倍数（不设上限）；k 在 HK 维、其余在 HV 维，
算子内部按 hk = hv // (HV/HK) 复用 key head，hm 与两个状态在 HV 维。
注意 GVA 只存在于 g-only（GDN）路径——gk 路径要求 HK == HV。

本轮范围：**GDN（g）+ KDA（gk）**。DPLR（gk + bg）**预留**——接口与 03 设计已覆盖
（S2 的 `bg^T @ V_c` 项、`M_c` 取 `+`、L1 的 `bg`/`V_c` 槽、`USE_BG` 分支都留着），
但本轮不进用例、不实现、不验收。

覆盖策略：A/B/D 三组把「形状维度」与「算法路径」做成叉乘，保证三条路径的条数接近；
GVA 组天然只落 GDN；负载组与性能组按需分布。

`risk` 是"预期不达标风险"（见 docs/design.md 4.3）：先精度后性能，但**每条用例都要满足
1.0x**，高风险项要在 04 阶段就预留优化手段。
"""
from __future__ import annotations

import json
from pathlib import Path

BT = 64
PATHS = ("GDN", "KDA")


def build_cases() -> list[tuple]:
    rows: list[tuple] = []

    # ★ 唯一的序列表达（2026-09-22 定稿，与竞品 CP 契约一致）：**varlen 打包窗口**
    #   B ≡ 1，cu_seqlens 必给；等长 batch 由调用方打包成等长多段。
    #   Nseq = 1 的单段窗口用 cu_seqlens=[0, T_win] 表达。

    # A 三路径 × 窗口规模：短 / 整块 / 长 / 超长 / 最小
    for path in PATHS:
        for T, risk, note in (
            (1, "高", "最小窗口：NT=1、尾块 1；固定开销主导"),
            (1023, "中", "短窗口 + 尾块：NT=16、尾块 63"),
            (4096, "低", "整块窗口：NT=64、尾块 0"),
            (8191, "低", "长窗口 + 尾块：NT=128、尾块 63"),
            (32767, "低", "超长窗口：NT=512，固定开销摊薄最好"),
        ):
            rows.append(("精度-窗口规模", f"T={T}", path, 1, 32, 32, T, [0, T], "FP32", risk,
                         note + "；单段窗口 Nseq=1"))

    # B 三路径 × 变长（多段打包，B = 1）
    for path in PATHS:
        for cu, tmax, risk, note in (
            ([0, 256, 512], 256, "低", "varlen 2 段等长：cu_seqlens=[0,256,512]，Nseq=2"),
            ([0, 512, 1024, 1536], 512, "低", "varlen 3 段等长：Nseq=3，校验 sequence 进 grid"),
            ([0, 64, 320, 1024], 704, "中", "varlen 3 段不等长且含 NT=1 的短段：Nseq=3"),
            ([0, 16387], 16387, "低", "varlen 单段对照：Nseq=1，T 非 64 倍数"),
        ):
            rows.append(("精度-变长", f"varlen {len(cu) - 1} 段", path, 1, 8, 8, tmax, cu, "FP32", risk, note))

    # C GVA（仅 GDN）：比例 + 与 varlen 的组合
    for hk, hv, note in (
        (16, 32, "GVA 1:2：HK=16、HV=32"),
        (21, 63, "GVA 1:3：HV 非 2 的幂"),
        (8, 32, "GVA 1:4：HK=8、HV=32"),
        (4, 32, "GVA 1:8：HK=4、HV=32"),
        (2, 64, "GVA 1:32：HK=2、HV=64，复用比例最大"),
    ):
        rows.append(("精度-GVA", f"GVA {hv // hk}:1", "GDN", 1, hk, hv, 2048, [0, 2048], "FP32",
                     "低", note))
    rows.append(("精度-GVA", "varlen + GVA 1:2", "GDN", 1, 16, 32, 8192, [0, 8192], "FP32", "低",
                 "varlen 与 GVA 组合：cu_seqlens=[0,8192]"))
    rows.append(("精度-GVA", "varlen + GVA 1:3", "GDN", 1, 21, 63, 16387, [0, 16387], "FP32", "低",
                 "varlen + GVA + 尾块：cu_seqlens=[0,16387]、HV 非 2 的幂"))

    # D 三路径 × gate dtype
    for path in PATHS:
        for dt, note in (("FP32", "gate 为 FP32"), ("BF16", "gate 为 BF16")):
            tag = "gk" if path != "GDN" else "g"
            rows.append(("精度-dtype", f"{tag} {dt}", path, 1, 8, 8, 1024, [0, 1024], dt,
                         "中" if dt == "FP32" else "低",
                         f"{tag} 为 {dt}" + ("（主要带宽项）" if dt == "FP32" else "")))

    # E 并行度（Nseq × HV）：覆盖"用不满核"到"多波"。GDN 表里的 (B, HV) 配对改写成
    # "把 B 条等长序列打包进一个窗口"（每段 4096，total = B × 4096），Nwork 不变。
    # 并行度与算法路径无关，这里按 GDN/KDA 交替取，避免本组把路径分布拉偏。
    # ⚠ gk（KDA/DPLR）路径要求 **HK == HV**（k 是已 gate 的 kg，见 docs/api.md），
    #   所以本组里 KDA 行的 HK 必须跟着 HV 走；GVA（HK < HV）只存在于 GDN 路径，
    #   其覆盖由 C 组（PPFM-19..25）保证。
    for b, hv, path, risk, note in (
        (1, 8, "GDN", "高", "Nwork=8：28 个 AIC 只用到 8 个，用不满核的最坏情形（GDN 表 V2 配对）"),
        (1, 16, "KDA", "高", "Nwork=16：只用到 16 个 AIC（GDN 表 V1 配对）"),
        (1, 32, "GDN", "低", "Nwork=32：1.14 波，刚好铺满（GDN 表 V3/V4 配对）"),
        (1, 64, "KDA", "低", "Nwork=64：2.29 波"),
        (16, 8, "GDN", "低", "Nwork=128：小 HV 靠 batch 补回并行度（GDN 表 C3 配对）"),
        (64, 8, "KDA", "低", "Nwork=512：18.3 波（GDN 表 C1/C6 配对）"),
    ):
        t_seg = 4096
        cu = [i * t_seg for i in range(b + 1)]
        hk = hv if path != "GDN" else 8      # KDA 必须 HK==HV；GDN 保留 GVA 复用
        rows.append(("精度-并行度", f"Nseq={b}×HV={hv}", path, 1, hk, hv, b * t_seg, cu, "FP32",
                     risk, note + f"；打包 {b} 段 × 每段 {t_seg}"))

    # F 子区间窗口（2026-09-23 定稿的竞品调用形态）：张量 T 轴比窗口长，
    #   cu_seqlens=[bos, eos] 的 bos 可 > 0、eos 可 < T；算子只算这一段。
    for path, cu, t_full, risk, note in (
        ("GDN", [40, 512], 512, "中",
         "子区间：张量 T=512，只算本地[40,512)（bos>0、472 token），对应竞品 contiguous 的 cu_seqlens[-2:]"),
        ("KDA", [444, 512], 512, "低",
         "子区间：张量 T=512，只算本地[444,512)（bos>0、68 token、NT=2），对应竞品 zigzag back 的 cu_seqlens[-2:]"),
    ):
        rows.append(("精度-子区间", f"sub-interval [{cu[0]},{cu[1]})", path, 1, 8, 8,
                     cu[-1] - cu[0], cu, "FP32", risk, note + f"；张量 T 全长 {t_full}"))

    # P 性能
    rows.append(("性能", "对齐 H20 model-gk", "KDA", 1, 32, 32, 11264, [0, 11264], "FP32", "低",
                 "H20 基线 case：决定 1.0x 判定的主口径"))
    rows.append(("性能", "对齐 H20 model-g", "GDN", 1, 32, 32, 11264, [0, 11264], "FP32", "低",
                 "g 路径的同一 shape"))
    rows.append(("性能", "CP=2 窗口 + GVA", "GDN", 1, 16, 32, 5632, [0, 5632], "FP32", "低",
                 "CP=2 下的真实窗口长度（11264/2）+ GVA 1:2"))
    rows.append(("性能", "长窗口", "GDN", 1, 32, 32, 16384, [0, 16384], "FP32", "低",
                 "长窗口：NT=256"))
    return rows


def build_rows() -> list[dict]:
    out = []
    for i, (group, src, path, B, hk, hv, T, cu, gdt, risk, note) in enumerate(build_cases(), 1):
        assert B == 1 and cu is not None, "唯一形态：B=1 + varlen 打包（cu_seqlens 必给）"
        segs = [b - a for a, b in zip(cu, cu[1:])]
        t_seg = max(segs)                      # 最长段决定 NT 与尾块（每条链独立推进）
        nt = -(-t_seg // BT)
        nseq = len(cu) - 1
        out.append({
            "用例ID": f"PPFM-{i:02d}",
            "分组": group,
            "来源": src,
            "算法路径": path,
            "layout": "BNSD",
            "B": B,
            "Nseq": nseq,
            "HK": hk,
            "HV": hv,
            "HV:HK": f"1:{hv // hk}",
            "K": 128,
            "V": 128,
            "T_win": T,
            "NT": nt,
            "尾块行数": t_seg - (nt - 1) * BT,
            "chunk_size": BT,
            "cu_seqlens": ("[" + ",".join(str(v) for v in cu) + "]") if len(cu) <= 5
                          else f"[0,{segs[0]},…,{cu[-1]}]（{nseq} 段等长，每段 {segs[0]}）",
            "varlen 打包": "是",
            "g/gk dtype": gdt,
            "Nwork": nseq * hv,
            "性能风险": risk,
            "覆盖点": note,
        })
    return out


if __name__ == "__main__":
    rows = build_rows()
    Path("cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} cases")
    by_path: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for r in rows:
        by_path[r["算法路径"]] = by_path.get(r["算法路径"], 0) + 1
        by_group[r["分组"]] = by_group.get(r["分组"], 0) + 1
    print("路径分布:", by_path)
    print("分组分布:", by_group)
    print("varlen 打包（B=1）条数:", sum(1 for r in rows if r["varlen 打包"] == "是"),
          "｜ Nseq>1 条数:", sum(1 for r in rows if r["Nseq"] > 1))
    print("风险分布:", {k: sum(1 for r in rows if r["性能风险"] == k) for k in ("高", "中", "低")})
