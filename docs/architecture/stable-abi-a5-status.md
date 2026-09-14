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
| `chunk_gated_delta_rule_fwd` **TND**（varlen 与 dense 拼法） | **未通过，未决** |

## 未决：fused forward 的 TND 拼法

- 触发条件：`layout="TND"`（BSND 同一组输入通过；TND 的 dense/varlen 都失败）。
- 现象：ctypes 参考成功，本 launcher 返回 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize failed: 161002`。
- 内核内的日志显示 A5 上该拼法会**内部转调 `ChunkGatedDeltaRuleFwdPrepare`**
  （`[ChunkGatedDeltaRuleFwdPrepare][Tiling] use_qk_l2norm currently must be true`）。
- 已排除：输出 shape/dtype（与参考逐一致）、format（4-D→NCHW、3-D→ND 已对齐）、
  storage shape（参考全部为 flat，已改为默认 flat）、标量参数（layout/scale/chunk_size/
  use_exp2/use_qk_l2norm/allow_neg_eigval/state_v_first 逐项一致）。
  参考的 9 个 descriptor 实测为：q/k/v `(1,128,4,128) format=0 storage=(65536,)`；
  g/beta `(1,128,4) format=2 storage=(512,)`；o `(1,128,4,128) format=0`；
  final_state `(2,4,128,128) format=0`；g_cumsum `(1,128,4) format=2`；
  A `(1,4,128,64) format=0`。
- 下一步方向：在 launcher 侧加一次性调试打印（或把 descriptor 序列化出来）逐个对拍，
  定位 161002 到底由哪个参数触发；也可以先按"A5 走 prepare+fwd_h+fwd_o 组合路径"实现，
  与参考在 A5 的实际行为对齐。

## 环境备忘

- 241 是 **x86_64**，221 是 aarch64：`.so` 不能跨机器复制，需在目标机上编译
  （A5 编译需要带 `torch/csrc/stable` 头文件的 torch，本机 conda `fzy` 的 2.7.1 没有这些头文件，
  用 `/usr/bin/python3.12` 的 torch 2.9 头文件 + 目标机的 libtorch 链接即可，运行在 2.7.1 上正常）。
- A5 OPP 构建：`FLA_NPU_SOC=ascend950 python3.12 scripts/build_wheel.py --wheel-dir <dir>`，
  需要可写的 `ASCEND_OPP_PATH`（否则 `opp/vendors/config.ini` 权限拒绝）；
  `cmake/third_party/build/modules/patch/*.patch` 必须随源码同步（打包时别用 `--exclude=build`）。
- 卡忙就换：`ASCEND_RT_VISIBLE_DEVICES=<n>`；设备 0 常年被他人占用且可能处于 Warning 状态。
