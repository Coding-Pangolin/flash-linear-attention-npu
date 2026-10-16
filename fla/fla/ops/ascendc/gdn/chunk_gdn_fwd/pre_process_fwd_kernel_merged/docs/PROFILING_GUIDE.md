# pre_process_fwd_kernel_merged 性能采集与分析指南（实测版）

> 适用：Ascend 950（950PR）上的 `pre_process_fwd_kernel_merged`（MIX 算子：1 AIC + 2 AIV），
> 也适用于同仓其它 AscendC 算子。所有命令与结论均在 **2026-09-28** 于
> **247（x86 + 真卡 950）** 与 **246（x86 + 真卡 950，容器 `admin123-gdn-test`）** 实测过。

---

## 0. 一页速查：三条采集路径，怎么选

| 目标 | 用什么 | 产物 | 实测状态 |
|---|---|---|---|
| **MindStudio Insight 看流水/热点** | **`msprof op`（msopprof 上板）** | `OPPROF_*/visualize_data.bin` + 各 `*.csv` | ✅ 247/246 都通 |
| 指令级时间线、空泡、无卡环境 | **`cannsim`**（CANN NPU Simulator；9.2+ 叫 `npusim`） | `npusim_*/report/results/kernel_*/core_*/trace_core*.json` | ✅ 247 通（x86 + 950） |
| 指令级 + 源码热点（无卡） | `msprof op simulator` + **aclnn 直调可执行程序** | `OPPROF_*/simulator/trace.json` + `visualize_data.bin` | ⚠️ 需绕开 torch_npu（§5） |

**重要结论（省时间）**
1. `msprof op simulator` 在本机（247/246, x86, CANN 9.1.0）**用 torch_npu 必然失败**：`aclInit 507000 / call acl_model init callback failed`。
   但**同一环境下纯 ACL 程序（`acl.init()+set_device`）能跑通** → **camodel 没问题，是 torch_npu 的 init 路径不兼容**。
   * **换卡没用**（失败在 aclInit；无设备容器、driver-less 容器、`LD_PRELOAD` camodel 库 都试过，一样挂）。
   * **换 CANN 包不一定解决**，要换就 torch/torch_npu + CANN 成对换；**更可靠的是绕开 torch_npu**（§5）。
2. 只要 Insight 可视化，用**上板** `msprof op` 就够了，而且 **Details/Cache 视图只有上板才有**。
3. `cannsim` 的 `trace_core*.json` 是 Chrome Trace，**可以导入 Insight 的时间线**（改名 `trace.json`），
   但**不产 `visualize_data.bin`**，因此 Insight 的**源码（Source）/详情（Details）/缓存（Cache）视图没有数据**。

---

## 1. 环境准备

### 1.1 机器 / 容器

| | 247 | 246 |
|---|---|---|
| SSH | `ssh admin123@192.168.13.247` | `ssh -o KexAlgorithms=diffie-hellman-group14-sha256 -o HostKeyAlgorithms=ssh-ed25519 -o IPQoS=none admin123@192.168.13.246` |
| 采集容器 | `admin123-ppfm-test`（privileged，CANN 9.1.0 自动 source） | `admin123-gdn-test`（privileged，CANN 9.1.0） |
| torch/torch_npu | 2.12.0 / 2.12.0 | 2.7.1+cpu / 2.7.1.post8 |
| 工作目录 | `/data/admin123/ppfm-sim`（容器内同路径） | `/data/admin123/bartonfang/ppfm-sim`（容器内 `/workspace/bartonfang/ppfm-sim`） |

> 246 的 SSH 必须带上面三个 `-o`（默认 KEX 会卡在 `KEX_ECDH_REPLY`）；
> 246 上 **`scp` 传大文件会被链路丢包**（实测 >1.3 KB 单包就断），用分块上传脚本（§7.4）。

### 1.2 让算子可被 import（二选一）

```bash
# A) 已 pip 安装 wheel 的容器（246 的 admin123-gdn-test 就是这种）
python3 -c "import fla_npu, fla_npu_opp_env; print('ok')"

# B) 不装、直接解包 wheel + PYTHONPATH（247 用的这种，避免动别人的环境）
W=/data/admin123/ppfm-sim
mkdir -p $W/wheel && cd $W/wheel
python3 -m zipfile -e $(ls -t /data/admin123/flash-linear-attention-npu/dist/*.whl | head -1) .
export PYTHONPATH=$W/wheel/flash_linear_attention_npu-26.7.0.dev0.data/purelib:${PYTHONPATH:-}
```

**两条铁律**
* harness 里 **`import fla_npu_opp_env` 必须早于 `import torch_npu`**（它把 wheel 内嵌 OPP 写进
  `ASCEND_CUSTOM_OPP_PATH`，CANN 初始化后再设就来不及）。
* **`PYTHONPATH` 只能追加**：`cannsim` 自己也是 python 工具（在 CANN 的 site-packages 里），覆盖掉会
  `ModuleNotFoundError: No module named 'cannsim'`。

---

## 2. 上板采集（主路径，出 `visualize_data.bin`）

### 2.1 harness 要求

* **单次调用、无 warmup 循环**（msopprof 自己会 replay）；shape 用真实模型规模最好（上板很快，不像仿真要小 shape）。
* 一个最小示例见 `sim_ppfm.py`（环境变量 `PPFM_SIM_T/HV/HK/VARIANT` 控制 shape 与 gdn/kda 路径）。
* 容器内不要设 `ASCEND_RT_VISIBLE_DEVICES`（真卡跑时按容器可见卡即可）。

### 2.2 命令

```bash
# 247（容器 admin123-ppfm-test；用解包 wheel 的 PYTHONPATH）
docker exec -it admin123-ppfm-test bash
export PYTHONPATH=/data/admin123/ppfm-sim/wheel/flash_linear_attention_npu-26.7.0.dev0.data/purelib:$PYTHONPATH
cd /data/admin123/ppfm-sim
PPFM_SIM_T=1024 PPFM_SIM_HV=8 msprof op \
    --application="python3 sim_ppfm.py" \
    --output=/data/admin123/ppfm-sim/out_msop_T1024_HV8 \
    --kernel-name=PreProcessFwdKernelMerged \
    --launch-count=1 --warm-up=1 --aic-metrics=Default
```

要点：
* `--kernel-name` 是**子串匹配**，用法名 `PreProcessFwdKernelMerged` 即可（真实符号是
  `PreProcessFwdKernelMerged_<hash>_1_mix_aic`）；不指定就会采到第一个 kernel（可能是辅助 kernel）。
* `--launch-count=1` 只采一个 kernel 实例；`--warm-up=1` 先跑一次。
* **多 shape 一把梭**：`bash run_msopprof_246.sh 64:1 256:2 1024:8 4096:8`（脚本里循环 + 最后自动分析）。

### 2.3 `--aic-metrics` 怎么选（实测）

| 传参 | 结果 |
|---|---|
| `Default` | ✅ **推荐**。产出 `PipeUtilization / ArithmeticUtilization / Memory / MemoryUB / MemoryL0 / L2Cache / ResourceConflictRatio / OpBasicInfo` + `visualize_data.bin`（~11 KB） |
| `Source` | ✅ 可用（单采，快速），`visualize_data.bin` 较小（只含源码热点相关数据） |
| `PipeUtilization,MemoryUB,TimelineDetail,Source,MemoryDetail`（组合） | ❌ **实测卡死**（几分钟无产出，只能 kill）——不要混着传，分开采 |

### 2.4 产出结构与"看什么"

```
out_msop_T1024_HV8/OPPROF_<ts>_<id>/
├── visualize_data.bin        ← 导入 MindStudio Insight（时间线 / 详情 / 源码 / 缓存）
├── OpBasicInfo.csv           ← kernel 名、Task Duration(us)、Block Dim、Mix Block Dim、频率
├── PipeUtilization.csv       ← 每个 block / 子核 各流水 busy 时间、占比、有效带宽
├── ArithmeticUtilization.csv ← cube/vector 指令耗时占比
├── Memory.csv / MemoryUB.csv / MemoryL0.csv / L2Cache.csv / ResourceConflictRatio.csv
└── dump/*.bin                ← 原始数据
```

**Insight 导入**：打开 MindStudio Insight → 左上「导入数据」→ 场景选**算子调优** → 选该 `visualize_data.bin`
（Insight 26.1 文档规定：算子调优可导入 `trace.json` 和 `visualize_data.bin`；其中
**时间线**←两者皆可、**详情/源码/缓存**←只有 `visualize_data.bin`）。
> 想看**源码（Source）热点图**：kernel 必须带调试信息 → 用 `python3 scripts/build_wheel.py -g` 重编再采。

---

## 3. 仿真采集（cannsim / npusim，指令级时间线）

### 3.1 硬约束

| 约束 | 说明 |
|---|---|
| 芯片 | 只支持 **Ascend950**（`-s Ascend950`） |
| 卡号 | 单卡且**不能改可见卡号**：脚本里 `unset ASCEND_RT_VISIBLE_DEVICES` |
| 范围 | **整个应用都被仿真**：host 侧每次 NPU 调用都会被模拟 → app 要极简 |
| shape | **必须小**：本算子 `T=64/HV=1`（~2.5 min）起步，最多 `T≤128`；单次调用、无 warmup |

### 3.2 命令（实测通过）

```bash
export PYTHONPATH=<wheel purelib>:$PYTHONPATH      # 只能追加！
unset ASCEND_RT_VISIBLE_DEVICES
cd /data/admin123/ppfm-sim
PPFM_SIM_T=64 PPFM_SIM_HV=1 cannsim record "python3 /data/admin123/ppfm-sim/sim_ppfm.py" \
    -s Ascend950 --gen-report -o /data/admin123/ppfm-sim/out_ppfm_T64_HV1_gdn
# 多核报告（默认只出 core0）
cannsim report -e <上一步的 npusim_* 目录> -n all -o <报告输出目录>
```

坑：
* `user_app` 必须是**一个字符串**（`"python3 xxx.py"`）；拆成两个参数报 `unrecognized arguments`。
* `--gen-report` 才有流水；不加只有 `record/`。
* 首次运行偏慢（op 包加载/冷启动，~2.5 min），同容器第二次同配置 ~1 min。
* 一次运行会检测到 **多个 kernel launch**（本轮 3 个：2 个辅助 + 1 个我们的）→
  **挑含 `AIC` 的那份报告**（辅助 kernel 只有 `AIV0/AIV1`）。

### 3.3 产出与查看

```
npusim_<ts>_sim_ppfm.py/
├── npusim.log                                   # 应用 stdout（[sim]/DONE 也会在这里）
├── record/instr.bin
└── report/results/kernel_<n>_reports/core_0/trace_core0.json[.gz]
```
* 图形化：把 `trace_core0.json` 拖进 `chrome://tracing` / https://ui.perfetto.dev（W/S 缩放、A/D 平移）。
* 命令行汇总：`python3 parse_ppfm_trace.py <report/results 目录>` → 每核×流水 busy 占比 + top 指令。
* trace 每条 `ph='X'` 事件字段：`name`(指令)、`cat/args.pipe`(流水)、`ts/dur`(cycle)、
  `pid/args.core_type`(核)、`args.pc`(pc 地址)、`args.instr_id`；**没有源码行**（要源码行见 §5）。
* 想导进 Insight：把 `trace_core0.json`（`.gz` 先解压）**改名 `trace.json`**，导入「算子调优」→ 只看**时间线**。

---

## 4. 分析手段（重点）

### 4.1 上板 CSV：先判"谁是关键路径、瓶颈在哪条流水"

工具：`python3 analyze_msopprof.py <T>:<HV>:<OPPROF目录> [更多 shape...]`

判读规则（按顺序）：

1. **并行度**：`OpBasicInfo.Block Dim` 与 `Nwork = Nseq × HV` 对比。`blockDim < 可用核数` ⇒ 核没吃满
   （本算子 `Nwork = Nseq×HV`，HV=8 单序列只起 8 个 block，而机器 20+ 核 → 3× 并行度被浪费）。
2. **谁是关键路径**：比较 `Task Duration` 与 AIC 的 `aic_time` / AIV 的 `aiv_time`。
   * `aiv_time ≈ duration` 且 `aic_time < duration` ⇒ **AIV bound**（本项目就是这种）。
   * 反之 ⇒ cube bound。
3. **瓶颈流水**：看主导子核（这里是 AIV）各流水占比。经验阈值：**某条流水 >50% ⇒ 它就是瓶颈**
   （小 shape 时本项目 MTE3 42–48%）；占比分散（20%~30% 多条）说明是**指令数/延迟**而不是单点带宽。
4. **带宽是否饱和**：CSV 里有 `*_active_bw(GB/s)`。与硬件上限比（HBM 数百 GB/s、L2 更高）：
   本项目 AIV MTE2 仅 15–35 GB/s、AIC MTE2 ~105 GB/s ⇒ **远未饱和 ⇒ 不是带宽瓶颈，而是搬运次数/同步/标量开销**。
   经验：`bytes ≈ active_bw(GB/s) × time(us) × 1000`，可反推每条流水的实际搬运量，与算法理论量对比找"多余搬运"。
5. **规模趋势**：至少采 2–3 个 shape，把 `duration` 对"每 block 的 chunk 数"做线性拟合 →
   得到 **固定开销 + 每 chunk 成本**（本项目：≈98 µs + 24.5 µs/chunk，四个 shape 全部吻合）。
   * 固定开销大 ⇒ 优化 prologue/launch；
   * 每 chunk 成本大 ⇒ 优化 chunk 内流水（见 §4.3）。
6. **指令构成**：`ArithmeticUtilization.csv` 给 cube/vector 指令耗时占比；`ResourceConflictRatio` 看冲突；
   `MemoryUB/MemoryL0/L2Cache` 分别看 UB/L0/L2 访问与缓存表现（L2 命中率低 ⇒ 搬运设计有问题）。

### 4.2 仿真 trace：看"具体哪条指令/哪个 pipe 在烧 cycle"

工具：`python3 parse_ppfm_trace.py <.../kernel_N_reports>`（聚合 pipe busy 占比 + top 指令），
细节用 Chrome/Perfetto 看时间线。

* 先确认挑对了 kernel 报告（含 `AIC` 的那份）。
* 关注：目标 pipe 的 busy 占比、`FC`（流控/`WAIT_FLAG`）占比大 ⇒ 空泡/串行等待、`PUSHQ` 大 ⇒ 指令发射压力。
* 上板与仿真**交叉验证**：上板告诉你"哪条流水 bound"，仿真告诉你"绑在哪些指令上"。

### 4.3 本项目当前画像（2026-09-28，GDN，上板 data）

| shape | 单元数 | duration | ns/单元 | AIV vec | AIV scalar | AIV MTE2 | AIV MTE3 | AIC CUBE | AIC fixpipe |
|---|---|---|---|---|---|---|---|---|---|
| T=64/HV=1 | 1 | 122.7 µs | 122744 | 11.9% | 17.2% | 14.3% | **42.6%** | 0.9% | 5.7% |
| T=256/HV=2 | 8 | 195.9 µs | 24485 | 17.5% | 21.1% | 14.9% | **33.5%** | 1.7% | 10.6% |
| T=1024/HV=8 | 128 | 489.0 µs | 3820 | 23.0% | 25.0% | 17.6% | 19.4% | 2.3% | 13.2% |
| T=4096/HV=8 | 512 | 1662.5 µs | 3247 | **25.8%** | **26.8%** | 17.5% | 13.0% | 2.5% | 14.8% |

结论：
* **AIV bound**（AIV elapsed ≈ task 时长），**CUBE 只有 0.9%→2.5%**，AIC 73%→88% 空闲。
* 成本模型 **≈98 µs 固定 + 24.5 µs/chunk**（prologue 的 128 行逐行初始化是那 98 µs）。
* 小 shape：**MTE3（UB→GM 写回）** 主导；大 shape：**VEC 25.8% + SCALAR 26.8%** 主导。
* 每核吞吐 38.5 单元/ms/block vs H20 44.6 单元/ms/SM ⇒ **单核只差 ~16%**，差距主要在
  **并行度（Nwork）** 与 **固定开销**。

优化优先级（用本指南验证前后差异）：
1. 状态更新去逐行标量（`ApplyStateUpdates` 每 rb 块 16 次 `Muls(...,GetValue)` → 1 次广播乘）；
2. h/m 常驻 UB + **UB→L1 直连**给 cube（去掉每 chunk ~192 KB 状态写回）；
3. **L0C→UB 直连**替 fixpipe→GM→AIV（AIC fixpipe 13–15% + AIV MTE2 17%）；
4. 提高 Nwork（多序列打包，用例表 PPFM-34/35 就是干这个）；
5. 压 prologue 98 µs（UB 造一次 h=0/m=I + 大块 DataCopy）。

---

## 5. 想要 `msprof op simulator`（无卡仿真 + 源码热点）怎么办

**实测事实**：本机 x86 上，`msprof op simulator` 跑 **纯 ACL** 程序成功（`ACL_OK`），
但跑 **任何 torch_npu** 程序都在 `torch.npu.set_device(0)` 处失败：

```
RuntimeError: ... aclInit, error code is 507000
E19999: Call rtRegTaskFailCallbackByModule("GeErrorTracking", ...) fail
       Assert ((RegErrorTrackingCallBack()) == ge::SUCCESS) failed [ge_executor.cc:322]
       call acl_model init callback failed
```

⇒ **camodel 没问题，是 torch_npu 的 init 路径与 camodel 不兼容**。因此：
* **换卡无用**（失败在 aclInit；driver-less 容器、无设备容器、`LD_PRELOAD` camodel 库都试过）。
* 换 CANN 包要**与 torch/torch_npu 成对换**，属于赌版本组合；
* **可靠做法（推荐）**：写一个 **aclnn 直调可执行程序**（不走 torch_npu），
  用 `msprof op simulator --soc-version=Ascend950PR_9581 --application="<可执行>" ...` 采集 → 得到
  `OPPROF_*/simulator/{trace.json, visualize_data.bin}`（含源码热点视图）。
  运算是 aclnn 接口：`aclnnPreProcessFwdKernelMerged`（头文件与 `libcust_opapi.so` 都在 wheel 里
  `fla_npu/opp/vendors/fla_npu_transformer/op_api/{include,lib}`）。

---

## 6. 常见坑／排错

| 现象 | 原因 / 处理 |
|---|---|
| `aclInit 507000 / call acl_model init callback failed` | torch_npu + camodel 不兼容（§5）。用纯 ACL 程序验证 camodel 是否正常 |
| `No module named 'cannsim'` | `PYTHONPATH` 被覆盖 → 改成追加 `$P:${PYTHONPATH:-}` |
| `unrecognized arguments: xxx.py` | `cannsim record` 的 `user_app` 要作为**一个字符串**传 |
| `No profiling data matched --kernel-name=...` | 名字没匹配到（用真实符号子串）或子进程真的失败（看 npusim.log / OPPROF 内有无文件） |
| 仿真卡死/无产出 | shape 太大或 app 里有多个重 kernel / atomic；降到 T=64、单次调用 |
| `--aic-metrics` 组合卡死 | 分开采（`Default`、`Source` 各自一次） |
| `--gen-report` 没有流水 | 没加 `--gen-report`（只有 `record/`） |
| 输出目录 750/root 导致宿主读不到 | 容器内 `chmod -R a+rX <out>` 或 `chown 1000:1000` |
| 246 `scp` 大文件失败 | 链路 MTU 问题，用分块上传（gzip+base64+分段追加，见 `remote_push.ps1`） |
| 246 SSH 卡在 `KEX_ECDH_REPLY` | 带 `-o KexAlgorithms=diffie-hellman-group14-sha256 -o HostKeyAlgorithms=ssh-ed25519 -o IPQoS=none` |

---

## 7. 脚本与产物清单

| 脚本 | 用途 |
|---|---|
| `sim_ppfm.py` | 采集用 harness（小 shape/单次调用；`PPFM_SIM_T/HV/HK/VARIANT` 控制） |
| `run_msopprof_board.sh` / `run_msopprof_246.sh` / `run_msopprof_shapes.sh` | 上板 msopprof 采集（单 shape / 多 shape + 自动分析） |
| `analyze_msopprof.py` / `analyze_all_ppfm.sh` | 分析上板 CSV：归一化到 per(chunk×head)、流水占比、搬运量估算 |
| `run_sim_ppfm_cannsim.sh` | cannsim 仿真采集（小 shape） |
| `parse_ppfm_trace.py` | 解析仿真 `trace_core*.json`：pipe 占比 + top 指令 |
| `probe_sim_246.sh` | 判定 `aclInit 507000` 归属（纯 ACL vs torch_npu 对照实验） |

产物位置（含 `visualize_data.bin`）：
* 247：`/data/admin123/ppfm-sim/out_msop_T{64_HV1,256_HV2,1024_HV8,4096_HV8}/OPPROF_*/`
* 246：`/workspace/bartonfang/ppfm-sim/out_msop_T{64_HV1,256_HV2,1024_HV8,4096_HV8}/OPPROF_*/`

参考文档：
* CANN《NPU Simulator 使用指南》`docs/zh/debug/npu_sim.md`（234 上有）
* 《算子调试调优》`docs/zh/debug/op_debug_prof.md`（234 上有）
* MindStudio Insight 用户指南（导入数据/算子调优数据说明）
* 221 上 cannbot skill：`ops/ops-simulator`、`ops/ops-profiling`、`code-performance-advisor`
