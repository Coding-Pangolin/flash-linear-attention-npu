# 全量算子薄层化：算子清单与批次表

> 基于 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 当前导出集合
> （#496 分支，未含 #390 的 causal_conv1d_update；后者合入后并入 conv1d 批次）。

## 迁移批次总览

| 批次 | 算子 | 说明 |
|---|---|---|
| A（已完成试点） | `npu_recurrent_gated_delta_rule` | thin 已启用、parity/多 stream 通过 |
| A | `npu_recurrent_kda` | recurrent 家族，in-place state |
| A | `npu_kda_gate_cumsum` | KDA 门控 cumsum（简单标量+数组） |
| B | `npu_causal_conv1d`（legacy）+ `causal_conv1d_update`(#390) | conv1d 家族；等 #390 合入统一 ABI |
| B | `npu_causal_conv1d_bwd` | 多输出 + char* layout，较复杂 |
| C | `npu_chunk_fwd_o`、`npu_chunk_gated_delta_rule_fwd_h`、`npu_chunk_fwd_h` | chunk GDN 主前向 |
| C | `npu_chunk_gated_delta_rule_fwd`、`npu_chunk_gated_delta_rule_fwd_prepare`、`_bwd_finalize`、`_bwd_dhu`、`npu_chunk_bwd_dv_local`、`npu_prepare_wy_repr_bwd*`、`npu_chunk_bwd_dqkwg` | chunk GDN 前后向细节，可选输出多 |
| C | `npu_chunk_local_cumsum`、`npu_chunk_scaled_dot_kkt`、`npu_recompute_w_u_fwd`、`npu_solve_tri` | chunk 工具算子；solve_tri 带 char* layout |
| D | `npu_chunk_kda_fwd`、`npu_chunk_kda_bwd`、`npu_chunk_kda_bwd_intra` | KDA 家族 |
| E | `npu_fast_gelu_custom`(+backward)、其余低频 | 收尾 |

## 分类口径

- **ND-only 可用**：输入都是普通 tensor/视图，无 5HD/NZ 风险 → 可进 thin；
- **参数形态复杂度**：
  - `简单`：全 tensor + 少量标量（int64/float）；
  - `中`：含 optional tensor / int-array / 输出 shape 推理；
  - `复杂`：多输出、char*、bwd、字符串 layout、多组 device+cpu metadata；
- **mutation**：`MUTATED_ARGUMENTS` 中列出的 in-place 参数需走 mutation 契约测试；
- **host 热度**：正式迁移顺序最终以服务 profile 决定；本表给出候选顺序。

## 详细清单

| 算子（python 名） | 批次 | 参数形态 | mutation | 备注 |
|---|---|---|---|---|
| npu_recurrent_gated_delta_rule | A | 简单-中（optional g/gk） | state | **已 thin** |
| npu_recurrent_kda | A | 中 | initial_state | recurrent 家族 |
| npu_kda_gate_cumsum | A | 简单 | - | KDA 门控 |
| npu_causal_conv1d（legacy） | B | 中（4 int-array） | conv_states | 等 #390 统一 |
| npu_causal_conv1d_update（#390 后） | B | 复杂（tensor+*_cpu 双通道/char*） | conv_state | 验证分支已实现原型 |
| npu_causal_conv1d_bwd | B | 复杂（char* layout、4 输出） | - | 需多输出 shape 规则 |
| npu_chunk_fwd_o | C | 中 | - | 主前向 |
| npu_chunk_gated_delta_rule_fwd_h | C | 中 | - | 主前向 |
| npu_chunk_fwd_h | C | 中（int-array cu_seqlens/chunk） | - | 纯 ctypes 入口 |
| npu_chunk_gated_delta_rule_fwd | C | 复杂（多可选输出） | - | Phase6 融合 |
| npu_chunk_gated_delta_rule_fwd_prepare | C | 复杂 | - | - |
| npu_chunk_gated_delta_rule_bwd_finalize | C | 复杂 | - | 多输出 |
| npu_chunk_gated_delta_rule_bwd_dhu | C | 复杂 | - | 多输出 |
| npu_chunk_bwd_dv_local | C | 复杂 | - | 多输出 |
| npu_prepare_wy_repr_bwd_full/bwd/da | C | 复杂 | - | 多输出 |
| npu_chunk_bwd_dqkwg | C | 复杂 | - | 多输出 |
| npu_chunk_local_cumsum | C | 简单 | - | - |
| npu_chunk_scaled_dot_kkt | C | 简单 | - | - |
| npu_recompute_w_u_fwd | C | 简单-中 | - | - |
| npu_solve_tri | C | 中（char* layout） | - | 需 char* 参数 kind |
| npu_chunk_kda_fwd | D | 中-复杂 | - | - |
| npu_chunk_kda_bwd | D | 复杂 | - | 多输出 |
| npu_chunk_kda_bwd_intra | D | 复杂 | - | 多输出 |
| npu_fast_gelu_custom / _backward | E | 简单 | - | 低频 |

> 精确迁移顺序最终以 host profile（调用次数 × 单次 host 时间）为准；
> 上表 A-E 为工程候选顺序，非最终发布顺序。

## 验证状态（持续更新）

| 算子 | JSON-only | 头校验 | parity | host（ctypes → thin public） |
|---|---|---|---|---|
| npu_recurrent_gated_delta_rule | ✅ | ✅ | 0.0 | ~0.5 → ~0.10 ms |
| npu_kda_gate_cumsum | ✅ | ✅ | 0.0 | 0.33 → 0.049 ms |
| npu_chunk_local_cumsum | ✅ | ✅ | 0.0 | 0.37 → 0.089 ms |
| npu_chunk_scaled_dot_kkt | ✅ | ✅ | 0.0 | 0.44 → 0.067 ms |
| npu_recompute_w_u_fwd | ✅ | ✅ | 0.0 | 0.58 → 0.115 ms |
| npu_prepare_wy_repr_bwd_full | ✅ | ✅ | 0.0 | 0.77 → 0.135 ms |
| npu_prepare_wy_repr_bwd | ✅ | ✅ | 0.0（KH=4/VH=8、bf16+fp32） | 0.74 → 0.129 ms |
| npu_chunk_bwd_dv_local | ✅ | ✅ | 0.0 | 0.48 → 0.105 ms |
| npu_prepare_wy_repr_bwd_da | ✅ | ✅ | 0.0 | 0.69 → 0.135 ms |
| npu_chunk_bwd_dqkwg | ✅ | ✅ | 0.0 | 0.73 → 0.097 ms |
| npu_fast_gelu_custom | ✅（base aclnn） | - | 0.0 | 0.26 → 0.041 ms |
| npu_fast_gelu_custom_backward | ✅（base aclnn） | - | 0.0 | 0.30 → 0.042 ms |
| npu_chunk_gated_delta_rule_fwd_h | ✅（v3 when/alloc） | ✅ | 0.0（dense/final/varlen） | 0.57 → 0.131 ms |
| npu_chunk_fwd_h | ✅（v3） | ✅ | 0.0（dense/final/varlen） | 0.63 → 0.127 ms |
| npu_chunk_fwd_o | ✅（v3） | ✅ | 0.0（仅 BNSD 合法域） | 0.56 → 0.097 ms |
| npu_chunk_gated_delta_rule_bwd_dhu | ✅（v3） | ✅ | 0.0（canonical ≥2 序列；单序列 dense 两条路径同 NaN，内核边界） | 0.71 → 0.139 ms |
| npu_recurrent_kda | spec 已建（inplace alias/return_when/python pre） | ✅ | 待实机用例：raw-gate canonical 组合（BSND + A_log + sigmoid）仍 507035 越界；仓库无本地测试资产，需拿到 design.md §11 对应测试工程或用 950 环境复现 | - |
| npu_chunk_gated_delta_rule_fwd_prepare | spec 已建（return_order + ctypes 委托） | ✅ | Ascend950-only（def 仅 AddConfig ascend950；910b 无 kernel config） | - |
| npu_causal_conv1d_bwd | ✅ | ✅（按文档签名） | 0.0（BNSD 域）；BSH/TND 两路径同 NaN（该构建 kernel 边界待查） | 0.58 → 0.084 ms |
| npu_chunk_kda_fwd | 未接入 | 头存在、def 支持 910b | ATK 用例全部 tag ascend950；当前 wave 未编 kernel（symbol 缺失），需先加 kernel 再自建 wrapper 合法 dense/BSND 配置 | - |
| npu_solve_tri | spec 已建（enabled=false） | - | 待修（thin 输出稀疏非有限，已回退 ctypes） | - |

## 下一步

已闭环 17 个算子（见验证状态表）。剩余算子及其所需能力：

1. `npu_recurrent_kda`（A）：需 codegen alias/out（inplace final_state 别名）+ zero
   initial_state 合成（python.pre 已支持）；仓库暂无现成 NPU 用例，需自建并覆盖
   BSND/TND × inplace × output_final_state。
2. conv1d 家族（B）：legacy `npu_causal_conv1d` 保持 ctypes 到 #390；`causal_conv1d_update`
   适配在验证分支（PR #512），#390 合入后并入；`causal_conv1d_bwd` 已按文档签名接入并
   在 BNSD 域验证，BSH/TND 的 NaN 属该构建 kernel 边界待查。
3. chunk 融合/准备（C）：`chunk_gated_delta_rule_fwd`、`fwd_prepare`（Ascend950-only，
   910b 无 kernel config）、`bwd_finalize`（Ascend950-only，910b 无法 parity）。
4. KDA chunk 家族（D）：`chunk_kda_fwd/bwd/bwd_intra`，布局矩阵
   BSND/BNSD/TND/NTD + canonical indices + workspace 分段策略。
5. `npu_solve_tri`：thin parity 待修（sparse/non-finite），spec 保持 disabled。
6. 收尾：上述算子全量迁移后执行 wheel 安装态全量回归 + 发布矩阵（Python 版本 ×
   linux x86/aarch64，wheel 含 cpXXX 平台标识属预期）。
