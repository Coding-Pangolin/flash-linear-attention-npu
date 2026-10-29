# Token-major g/beta 迁移开发记录

状态：开发中，不能作为最终设备验收。

用户确认 g/beta 固定 token-major，qkv layout 独立。方案变更为 Stage0 的输入地址、DMA stride、beta UB 解包和索引初始化；Stage1–6 的数学、BF16 relay、资源及跨核协议保持既有实现。

CPU 标杆数学/比较器不变，AST 检查确认只有两个输入解析表达式变化。纯 CPU 96 组合覆盖四 layout、六 key、四 dtype，两套参考共比较1824输出；仅6个 control beta_eff 输出因 CPU sigmoid 的连续/strided vector 路径出现最多1.1920928955078125e-7差值，其余输出逐位相同。这是标杆解析核对，不代替原生 NPU 双标杆，也不改变正式容差。

首个独立 Prepare 候选待 clean 源编译、小边界原生比较和错误注入，再测 case1/31 与四模型。完整 forward/V2 的公共校验及 fused A5 beta adapter 尚待同步，不得将本阶段独立 Prepare 候选宣称为端到端可交付版本。

已知数值问题：基线1ff的部分 FP32 Precomputed gk 前缀未通过冻结标准；TwoSum 独立实验通过补充精度但具有性能成本，本次地址迁移没有自动合入该数学变化。最终交付必须解决并重新验证。

最终验证范围：新合同全32/416、紧凑泛化与同规模组合、六 key MSS/DC、实际对象/key 绑定、全成本 ABBA5/50/3及A5匹配证据。资源仍为 AIV UB190KiB、两代 GM slot 与原十个跨核 flag；运行验证待补充。
