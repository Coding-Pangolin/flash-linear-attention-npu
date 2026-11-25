# KcpAllGatherMerge

## 功能

实验性原生通信边界：一次 MIX launch 执行 FP32 AllGather，再按 rank 顺序合并
exclusive prefix，供 [KDA CP 整前向](../chunk_kda_cp_fwd/README.md) 使用。
AllGather 完成后才执行依赖它的 Merge；核内单轮通信与本轮 Merge 不重叠。

## 输入

| 参数 | dtype / shape | 约束 |
|---|---|---|
| summary | FP32 `[H,128,256]` | 连续、公开格式；最后一维是 `[E,M]` |
| h | FP32 `[H,128,128]` | 原地输出；rank 0 的调用方须预置零 |

## 输出

返回 `(h, gathered)`；`h` 与输入同一张量，`gathered` 是 FP32 `[P,H,128,256]`。
所有 rank 都参与通信；rank 0 保留零状态，rank r 仅应用 `[0,r)` 的摘要。

## 属性

`group` 为已初始化的 HCCL communicator 名称；`rank_size=P`、`rank=r` 为该域内
大小和 rank。HCCL 名称查询使用全局 rank，前缀计算使用组内 rank，两者不可混用。

## 支持范围

H=1..16、P=2..8、K=V=128、FP32、公有连续存储。A2 使用 AICPU，A5 编译 CCU
候选。A2 已有 CP2/CP4 通信及 prefix 门禁证据；A5 原生资源注册尚未通过，整前向
`mc2` 路径目前明确限制为 A2。详见 [验证记录](../chunk_kda_cp_fwd/docs/validation.md)。

## 已知限制

本入口无自动回退、无反向、无 packed/zigzag。调用方必须保证全域元数据及调用顺序
一致；单进程 ACL 参数校验无法证明通信域其他成员的状态。A5 CCU 的失败发生在
资源注册阶段，不能据此宣称 FP32 CCU 通信不受支持。正式全平台验收尚未完成。

接口与返回码见 [api.md](docs/api.md)，资源和同步见 [design.md](docs/design.md)。
通信阶段参考官方 AllGatherAdd（ops-transformer 155e4400，CANN OSL v2）；Merge
引用本仓 BSD 实现，构建时将依赖头复制到构建目录，未复制一份算子源码到工作树。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_BUFFSIZE=32 \
torchrun --standalone --nproc_per_node=2 examples/cp/check_kcp_native.py \
  --heads 3 --output outputs/native_cp2
```
