# Phase6 私有算子实现

本目录保存 `ChunkGatedDeltaRuleFwd` 在单 kernel 内直接复用的实现代码。它们不是独立算子入口，也不参与对应公共小算子的注册、ACLNN 接口或单算子构建。

当前包含：

- `chunk_local_cumsum`：门控累加实现；
- `recompute_w_u_fwd`：W/U 重计算实现及其私有 tiling 数据处理；
- `chunk_gated_delta_rule_fwd_h`：状态更新 H 实现和 H→O 流水同步；
- `chunk_fwd_o`：输出 O 实现和 H→O 流水消费逻辑。

这些文件由 Phase6 固定版本的实现复制而来。融合专用的任务映射、跨核同步和流水模板只在这里维护；对应公共单算子目录保持固定 main 基线。修改本目录后必须重新执行 Ascend950 全量构建、模型场景 raw-bit 一致、实际 kernel 路由和性能无稳定劣化门禁。
