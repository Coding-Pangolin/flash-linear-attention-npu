# A2 Prepare：六 key 原生 ATK 验收

NPU DUT 只调用已安装包的公开 `chunk_kda_fwd_prepare`。正式入口现场生成
公开输入、计算两路 CPU 参考，不依赖本地 tensor、参考或 DUT 输出缓存。
本文描述验收合同；不声明当前包已通过。逐项结果随最终包独立归档。

## 范围与身份

A2 key 仅包含 GateT、BetaT、pipelineVariant。数学/输出模式为一次调用内
不可变的 tiling 字段。AIC/AIV 共用同一派发，BF16 relay 优先，
其次是 CommonMath，最后是运行时泛化路径。A5 保留旧入口。

| key | gate / beta | pipeline | MSS/DC id |
| ---: | --- | --- | ---: |
| 2570 | BF16 / BF16 | Generic | 0 |
| 7690 | BF16 / FP32 | Generic | 1 |
| 2590 | FP32 / BF16 | Generic | 2 |
| 7710 | FP32 / FP32 | Generic | 3 |
| 68126 | FP32 / BF16 | Target | 4 |
| 68106 | BF16 / BF16 | Target | 5 |

这些是 CANN 编码器预期值；必须另采实际 key。Generic 代表为 H5/T1665，
Target 保持 H96/T8192 合同。默认 DC/MSS 每 key 一例，不重复追加 target 文件。
A5 的原432例保存在 `_a5_legacy_mss.json`，不可把 A2 检查推断为 A5 通过。

## 输入、参考与输出

21个公开输入及13个输出槽位保持原顺序；禁用输出必须为 None。
none/recompute/save/forward 分别比较6/12/13/7个启用输出，
不能通过过滤错误输出、修改 dtype 或缩小 shape 取得通过。

三个 ATK 角色使用 `--syc_dataset` 的同一份已量化输入：

- NPU DUT：每轮执行新安装包，检查全部启用输出的合同及有限值。
- truth_fp64：benchmark CPU 角色，FP64完整求逆；不模拟 DUT 的近似和内部 BF16 舍入。
- control_fp32_bf16：非 benchmark CPU 角色，FP32计算，保持原参考定义中的 BF16 舍入点。

两个 CPU 角色共享已审计数学与布局转换，并非两个独立来源的算法。
有限值检查在 CPU 上读取每个完整 DUT 输出，返回给 ATK 的 NPU tensor
及 dtype 保持原样，避免额外的 Cast/IsFinite/ReduceAll 干扰 MSS 插桩。
使用原生 `cv_fused_double_benchmark`：max=5、avg=1.5、RMS=1.5，
其他默认项不改。要求原生比较器可导入；不得以本地补丁替代缺失比较器。

## 固定清单

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_prepare.json` | 200 | 完整原始输入与 id/name/seed；保留原有 Save 输出 |
| `atk_chunk_kda_fwd_prepare_forward.json` | 4 | 独立补充四布局 Forward，不替换原始200例 |
| `atk_chunk_kda_fwd_prepare_perf.json` | 128 | 32模型 × 普通/safe gate × FP32/BF16 gate |
| `atk_chunk_kda_fwd_prepare_mss.json` | 6 | 每个实际 key 的 DC50及四种 MSS |
| `atk_chunk_kda_fwd_prepare_paths.json` | 10 | 开发专项：尾块、奇数 head、ring/drain与回退 |
| `atk_chunk_kda_fwd_prepare_regression_perf.json` | 10 | 保留旧性能护栏 |
| `runtime_minimal.json` | 4 | Generic 最小精度快筛；不代替完整验收 |

模型清单保留完整 cu_seqlens，默认 seed3。beta BF16、sigmoid=true、
allow_neg=false、exp2=true、L2、scale=1/√128、save。这些尚未确认的模型属性
仍是显式测试配置。200泛化及128模型各只跑原 seed，不默认扩成600/384。
额外种子仅由 `--seeds` 显式启用，不能计作更多泛化结构。
精度可显式设置 `--precision-jobs 3` 并行独立 case；默认1。
DC/MSS/performance始终串行，性能配对期间不得并行运行精度。

## 正式入口

先加载 CANN、官方 ATK 和隔离安装包的环境，核对真实导入路径和 op API。
所有输出目录必须全新；脚本拒绝覆盖旧报告。

```bash
python -B scripts/test_delivery.py
python -B scripts/delivery_cases.py
python -B gen_chunk_kda_fwd_prepare.py --summary
atk case -f chunk_kda_fwd_prepare.yaml -p gen_chunk_kda_fwd_prepare.py
python -B scripts/audit_generated.py result/chunk_kda_fwd_prepare/json/all_chunk_kda_fwd_prepare.json

python -B scripts/run_delivery.py --scope accuracy --out /tmp/prepare_acc_new \
  --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/exact.whl --kernel-lib /path/to/libcust_opapi.so
python -B scripts/run_delivery.py --scope model-accuracy --out /tmp/prepare_model_new \
  --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/exact.whl --kernel-lib /path/to/libcust_opapi.so
python -B scripts/run_delivery.py --scope accuracy --out /tmp/prepare_forward_new \
  --case-file atk_chunk_kda_fwd_prepare_forward.json \
  --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/exact.whl --kernel-lib /path/to/libcust_opapi.so
python -B scripts/run_delivery.py --scope determinism --out /tmp/prepare_dc_new \
  --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/exact.whl --kernel-lib /path/to/libcust_opapi.so
```

DC 固定原生50轮，不能用独立逐位诊断代替。MSS 使用同源码的插桩包，
四种工具分别执行；以下命令对 memcheck/racecheck/initcheck/synccheck 各运行一次：

```bash
python -B scripts/run_delivery.py --scope mssanitizer --tool memcheck \
  --out /tmp/prepare_mem_new --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/instrumented.run --kernel-lib /path/to/instrumented/libcust_opapi.so
ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_MODULE_LOG_LEVEL=OP=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
python -B scripts/run_delivery.py --scope route-trace --out /tmp/prepare_keys_new \
  --device 0 --port 19494 --source-commit EXACT_COMMIT \
  --package /path/to/exact.whl --kernel-lib /path/to/libcust_opapi.so
python -B scripts/summarize_run.py /tmp/prepare_acc_new
python -B scripts/audit_mss.py /tmp/prepare_mem_new --tool memcheck \
  --manifest /path/to/instrumented_build_manifest.json
```

精度入口检查原生每 case 状态、输出数量和每输出比较，不只看退出码。
DC/MSS/perf 退出0只标记 reports_pending_review；仍须审核真实 key、
自定义 kernel 的插桩 start/finish、工具错误和完整报告。超时、缺行、
晚到告警或部分通过均不算完成。route-trace 仅证明派发，不证明精度。
`audit_mss.py` 流式检查原始等级、逐 key 告警及完整日志哈希；六个真实 key
各一次插桩 start/finish 必须完整对应。ERROR 保留为未关闭门禁；只有 WARNING
时标为 `warnings_pending_review`，不直接认定算子错误，也不自动豁免。GM 所有权
提示、冗余等待和寄存器提示应分别结合实际访问范围、依赖链及同输入旧版对照
甄别。零 ERROR/WARNING 且 native 完整通过才自动标为 `instrumented_native_pass`。
即使 ATK 列为 True，缺插桩、未甄别的晚到告警或包身份不一致仍不能宣称完成验收。
性能测量不打开 OP 调试日志，并使用最佳版本同窗口 ABBA 对照；
Task Duration 与 Event 分列，至少 warmup5、3×50及两个独立窗口。

## Host 合同测试

从仓库根目录编译运行 `scripts/test_target_tiling.cpp`、
`scripts/test_relay_tiling.cpp`，以及 op_host/tests 中的调度测试。
`scripts/test_runtime_keys.cpp` 需要 CANN include，用共享声明检查六 key。
`scripts/test_tiling_wire.cpp` 需要 host/kernel include及 CANN register 库，
用真实 CANN SaveToBuffer逐字段核对72字节序列化；启动前加载 CANN 环境。

归档 source/patch、wheel/run、所有 kernel、API、Python wrapper、执行器、
比较器及输入清单哈希。测试报告和大二进制放独立实验目录，不提交本目录。
日常可用已有参考缓存加速筛查，但缓存结果不能冒充上述正式无缓存验收。
