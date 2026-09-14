# 热路径 host 开销：本方案 vs vLLM-Ascend（2026-09-14，910B3）

环境：221，`env_full`（OPP 由 main 现编），launcher `libfla_npu_thin_c5.so`，卡 4。
两侧跑同一个对比脚本（`bench_recurrent_variants.py` / `bench_conv1d_vllm.py`），
形状与计时口径一致：host 时间是 Python 调用把活交给 aclnn 的墙钟时间（计时区间内
不 synchronize），device 时间是 NPU event。

## recurrent GatedDeltaRule：batch 128、q=1、非连续 state、100 次

| 路径 | host P50 | host mean | device P50 |
| --- | --- | --- | --- |
| FLA stable launcher | **0.1665 ms** | 0.1962 | 0.2776 ms |
| FLA ctypes（参考实现） | 0.5838 ms | 0.6486 | 0.2797 ms |
| vLLM-Ascend custom | **0.1132 ms** | 0.3736（p99 3.80） | 0.2708 ms |

- launcher / ctypes = **0.29×**（快 3.5 倍），差的就是"描述符森林"。
- launcher / vLLM = **1.47×**（同一量级），device = **1.03×**（同 kernel）。
- vLLM 的 host 分布长尾明显（mean 0.37、p99 3.8），我们是紧的（mean 0.20、p99 0.61）；
  比较应看中位数。

## causal_conv1d update：batch 100、连续 state、200 次

| 路径 | host P50 | device P50 |
| --- | --- | --- |
| FLA stable launcher（`out=`） | **0.1797 ms** | 0.2568 ms |
| vLLM-Ascend custom | 0.2289 ms | 0.3947 ms |
| FLA ctypes（参考实现） | 0.6412 ms | 0.7388 ms |

- launcher / vLLM = **0.79×**（host 更快）；device 差异来自两侧用的是不同 conv1d kernel
  （FLA OPP vs vLLM custom_transformer），不是适配层。
- launcher / ctypes = **0.28×**。

## 结论

- vLLM 实际调用的两个算子，现在 host 开销都处在 vLLM 同一量级或更好；被替代的
  ctypes 路径要贵 3.5–8.6 倍。
- recurrent 残留的 1.47×（P50）来自 boxed 调度 + 公共 wrapper（迁移计划的 R19 记录项），
  device 侧比值约 1.0，说明与 kernel 无关。
- 机器负载会移动绝对值；同一次运行里三条路径（同一脚本、同一形状）的比值才是可比的。
