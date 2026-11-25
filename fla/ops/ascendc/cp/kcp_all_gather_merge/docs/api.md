# KcpAllGatherMerge API

## Python 主入口

```python
from fla_npu.ops.ascendc import kcp_all_gather_merge
h, gathered = kcp_all_gather_merge(summary, h, group, rank_size, rank)
```

`npu_kcp_all_gather_merge` 是相同实现的别名。输入、输出和属性唯一规格见
[README](../README.md)。Python 将 communicator 名编码为 CPU INT64 字节张量，
Stable adapter 恢复字符串后调用 aclnn；CPU 张量不参与设备计算。

调用前，所有组员须验证相同 H/P 和唯一组内 rank，且按相同顺序进入通信。
建议通过 `create_kda_cp_plan` 使用整前向入口完成集体预检。裸接口只能校验本地值。
调用流可为非默认 NPU stream；调用方负责跨流依赖和分配器生命周期。

## aclnn

```cpp
aclnnStatus aclnnKcpAllGatherMergeGetWorkspaceSize(
    const aclTensor *summary, const aclTensor *h, const char *group,
    int64_t rankSize, int64_t rank, const aclTensor *gathered,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnKcpAllGatherMerge(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

`h` 被传给内部 h0 输出，属于原地写入；rank 0 必须预置为零。`gathered` 由调用方
分配且与 summary/h 不重叠。所有 descriptor 为连续公开格式；summary 的 storage
shape 也须保留 `[H,128,256]`。workspace 必须采用第一阶段返回值，不能仅分配 Merge
scratch。executor/descriptor 生命周期遵守标准两阶段 ACLNN 契约。

## 返回码

| 场景 | 结果 |
|---|---|
| 成功 | `ACLNN_SUCCESS` |
| 必选 tensor、group、workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| 空 group、本地 P/r 越界、shape/dtype/连续性/format 不匹配 | `ACLNN_ERR_PARAM_INVALID`，附结构化参数信息 |
| CANN 无所需 MC2 服务接口 | `ACLNN_ERR_INNER`，附服务错误信息 |
| 通信域不存在、服务分配或运行时失败 | 传播 CANN/HCCL 错误；禁止静默改走框架通信 |

服务选择、错误状态和构建适配不是性能保证；每个目标设备须独立通过通信门禁。
