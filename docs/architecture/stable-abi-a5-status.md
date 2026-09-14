# Ascend950 验证状态（2026-09-14）

机器：`sz-blue-950pr-13-241`（x86_64 + Ascend950PR_9579），CANN 用
`/home/npu_user7/lizhuo92/cann/cann-9.1.0`（可写的用户侧 CANN），launcher 由
`/usr/bin/python3.12`（torch 2.9 头文件）编译，测试跑在 conda `fzy`（torch 2.7.1）。
OPP 为本分支 main 现编的 A5 wheel（`flash_linear_attention_npu-26.7.0.dev0-950.x86_64-py3-none-any.whl`）。

## 结论

| 检查 | 结果 |
| --- | --- |
| A5 OPP 头文件 ↔ 调用点（`op_abi_validate.py`） | **47 call site，0 mismatch**（只有 `aclnnSolveTri` 属已知无公开头文件） |
| launcher 加载 + 构建戳 | `_stable.available() == True`（用 2.9 头文件编译，跑在 torch 2.7.1 上） |
| `chunk_gated_delta_rule_fwd_prepare`（950-only） | 6/6 PASS |
| `chunk_gated_delta_rule_bwd_finalize`（950-only） | PASS |
| `chunk_gated_delta_rule_fwd` BSND（含 state_v_first / h 输出） | 3/3 PASS |
| `chunk_gated_delta_rule_bwd`（新融合反向，A5-only） | 由 `--group a5` 覆盖（本次会话新增场景） |
| `chunk_gated_delta_rule_fwd` TND | PASS（见下"TND 修复"） |
| `chunk_gated_delta_rule_bwd`（融合反向） | 记录为限制：该 OPP 上参考实现自己也拒绝（161002），8 种配置全试过 |

## TND 修复：rank-4 拼法读成了 rank-3

`layout="TND"` 在 A5 上返回 161002 的根因是**适配层的 shape 计算**：这个算子接受
TND/NTD 这两个名字，但**始终按 rank-4 读**（TND 的 token 轴是 dim 1，head 是 dim 2，和
BSND 一样）。我们用了 packed（rank-3）helper，于是四个输出全部算错——
用 `FLA_STABLE_DEBUG_DESC=1` 打出来是 A `[1,128,1,64]`、final_state `[2,128,4,4]`，
而参考是 A `[1,4,128,64]`、final_state `[2,4,128,128]`。改成 rank-4 helper
（`tokens4`/`value_heads4` + `size_of(q,3)`/`size_of(v,3)`）后 TND 通过。

顺带确认了 descriptor 约定（两种实现现在逐字段一致）：contiguous 张量默认 **flat
storage**（`(numel,)`），4-D 的 format 是 **0（NCHW）**、3-D 是 **2（ND）**；
`nd_tensor`/`logical_tensor` 只用于参考确实覆盖的那 9 处调用点。

## 950 上跑全量矩阵的结果

- A5 专用组（`--group a5`）：**12 PASS / 4 记录限制**（fwd_prepare 6、bwd_finalize 1、
  fused fwd BSND×3+TND+BNSD 5；融合反向 2 条"两边都拒绝"+2 条 flag 拒绝）。
- 950 跑**完整矩阵**：**276 PASS / 16 SKIP**，停在一个未决用例：
  `chunk_gated_delta_rule_fwd(varlen_B1_T128_c64)` 的 **output[3]** 与参考不一致
  （diff 3.4e38，即未写入区域当有效值比较）。这是 BNSD varlen 拼法在 950 上的
  行为差异，需要下一步定位（A2 上该用例是逐位一致的）。
- **kernel 缺陷（记录，供 OPP 侧修）**：`aclnnChunkKdaBwdRecompute` 在
  `use_gate_in_kernel=False` 时抛 **AI Core exception（错误码 271）** 并把设备带进错误
  状态；`chunk_kda_bwd_recompute(no gate)` 因此改为记录而不执行。

## 环境备忘

- 241 是 **x86_64**，221 是 aarch64：`.so` 不能跨机器复制，需在目标机上编译
  （A5 编译需要带 `torch/csrc/stable` 头文件的 torch，本机 conda `fzy` 的 2.7.1 没有这些头文件，
  用 `/usr/bin/python3.12` 的 torch 2.9 头文件 + 目标机的 libtorch 链接即可，运行在 2.7.1 上正常）。
- A5 OPP 构建：`FLA_NPU_SOC=ascend950 python3.12 scripts/build_wheel.py --wheel-dir <dir>`，
  需要可写的 `ASCEND_OPP_PATH`（否则 `opp/vendors/config.ini` 权限拒绝）；
  `cmake/third_party/build/modules/patch/*.patch` 必须随源码同步（打包时别用 `--exclude=build`）。
- 卡忙就换：`ASCEND_RT_VISIBLE_DEVICES=<n>`；设备 0 常年被他人占用且可能处于 Warning 状态。
