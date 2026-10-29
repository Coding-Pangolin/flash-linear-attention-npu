# A2 Prepare 原布局合同恢复验证

当前开发范围：仅 A2 Prepare 优化；q/k/v/g/beta 全部跟随 layout。
原 A5、完整 forward/V2、公共 wrapper、Example 与其参考的合同迁移改动已撤回。
六key、BF16 relay、数值修复、任务/资源/事件不变；仅恢复来源地址和分块读取策略。

恢复前源码 ab844e58 的200/2096与数学基线81fc9dac的32/416等证据均属历史合同，
不能标记为当前恢复合同的通过结果。当前源码 clean 构建、小例/四模型、原200/32、
六key内存/DC50、逐例ABBA与外围兼容检查待执行。冻结双标杆和容差不变。

来源参考：仓库 CHUNK_INDEPENDENT V1，a78efa9512dfa64c20b2a58a75e84e88a054d694；
复用任务/事件工程边界，不改变本轮流水生命周期。已评审设计见 design_a2.md，
恢复后的唯一公开合同见 api.md。原 A5 design.md 保留。
