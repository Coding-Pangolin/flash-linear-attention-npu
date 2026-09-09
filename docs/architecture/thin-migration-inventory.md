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
| npu_chunk_local_cumsum | ✅ | ✅ | 0.0（910b w16 + 950 env950c/run merge） | 0.37 → 0.089 ms |
| npu_chunk_scaled_dot_kkt | ✅ | ✅ | 0.0（k fp16/bf16 × g/beta fp32；OPP 仅编译该 dtype 域，fp16 g/beta 为 161002） | 0.44 → 0.067 ms |
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
| npu_recurrent_kda | ✅ | ✅ | 0.0（BSND B2/T2/H2/HV4 dense、state_v_first、inplace + final_state；Ascend950PR） | 0.085 → 0.011 ms（950） |
| npu_chunk_gated_delta_rule_fwd | spec 已建（BSND 域 + return_order None 槽） | ✅ | 运行域待定：910b/950 多组 flag 探针均 161002（host 走 l0op phase6 复合，可能依赖特定子算子/flag 组合）；仓库无调用方 | - |
| npu_chunk_gated_delta_rule_fwd_prepare | ✅ | ✅ | 0.0（9 输出；Ascend950PR） | 0.144 → 0.030 ms（950） |
| npu_chunk_gated_delta_rule_bwd_finalize | ✅ | ✅ | 0.0（5 输出；Ascend950PR，g/beta fp32、G=2 域） | 0.167 → 0.023 ms（950） |
| npu_causal_conv1d_bwd | ✅ | ✅（按文档签名） | 0.0（BNSD 域）；BSH/TND 两路径同 NaN（该构建 kernel 边界待查） | 0.58 → 0.084 ms |
| npu_chunk_kda_fwd | ✅（dense BSND 合法域；其它布局/flag 委托 ctypes） | ✅ | 0.0（10 输出 + None 语义） | 1.04 → 0.114 ms |
| npu_chunk_kda_bwd_intra | ✅（BNSD dense 单发射合法域；BSND 分段路径委托 ctypes） | ✅ | 0.0（4 输出） | 0.73 → 0.091 ms |
| npu_chunk_kda_bwd | ✅（dense BNSD 简单域：偶数头、T%64=0、gate off；tail/奇头/varlen 回退委托 ctypes） | ✅ | 0.0（dq/dk/dv/db/dg + 3×None） | 0.88 → 0.111 ms |
| npu_solve_tri | ✅（dense bsnd/bnsd 域，enabled） | ✅ | 0.0（fp16/bf16 × BT 16/32/64/128，910b w16；950 env950c/run merge ext 直连亦 0.0）；TND/NTD varlen thin 仍非有限 → 委托 ctypes | 0.372 → 0.098 ms（910b w16，dense bsnd fp16 BT64） |

## 下一步

已闭环 24/26 个库存算子（见验证状态表；count 不含 conv1d legacy 与
运行域未定的 composite fwd）。剩余算子/子域及其状态：

1. `npu_causal_conv1d`（legacy）：保持 ctypes 到上游 #390 合入后统一 ABI。
2. `causal_conv1d_update`（#390）：适配已在验证分支 PR #512 实现并跑通原型；
   #390 合入后并入本分支并做 910b/950 全量回归。#390 当前（2026-09-09）仍 open，
   最新 head 与本分支差异仅 examples/flash_gated_delta_rule.py，ABI 未变。
3. `npu_chunk_gated_delta_rule_fwd`（composite）：spec/适配已建（BSND 域 +
   return_order None 槽），但 910b/950 多组 flag 探针均为 161002，仓库无调用方；
   判定为运行域未闭合，待真实调用/子算子组合确认后回归。
4. `npu_solve_tri` varlen（TND/NTD）：thin 直连结果非有限 → wrapper 已委托
   ctypes；dense bsnd/bnsd 已原生 thin 并多处验证 0.0。
5. 收尾：regression_thin_ops 20 场景（37 组）已在 910b 最新 wheel（w16）上
   全量执行并全绿；950 侧通过 env950c + 2 算子 OPP run 合并验证了
   chunk_local_cumsum（回归组 0.0）与 solve_tri dense（ext 直连 0.0），其余
   通用与 950-only 场景全绿；整 wheel 重编四次因共享机 /home 长期 100%（可用
   <2G，kernel 编译期 ENOSPC）中断，待磁盘恢复后补跑完整 20 场景；发布矩阵与
   实测记录见 thin-launcher-release-matrix.md。
