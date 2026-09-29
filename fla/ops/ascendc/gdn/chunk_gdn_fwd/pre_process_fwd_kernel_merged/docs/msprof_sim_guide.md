# 算子仿真流水采集指南（实测版，2026-09-28）

目标：给 `pre_process_fwd_kernel_merged`（以及同类 AscendC 算子）采**指令级仿真流水**，
定位 AIC/AIV 各流水占比与热点指令。

**实测结论（先说结果）**：在 **247 / 容器 `admin123-ppfm-test` / CANN 9.1.0 / x86_64 / Ascend950** 上
* `cannsim`（CANN 自带 NPU Simulator，9.2+ 改名 `npusim`）**可用** —— 本指南主路径；
* `msprof op simulator`（MindStudio 的仿真模式）**在本机起不来**，卡在 `aclInit 507000`（见 §6），
  不要在这台机器上浪费时间；需要它时走 aarch64 + `LD_PRELOAD` 的路线（§6）。

---

## 1. 硬约束（踩了就白跑）

| 约束 | 说明 |
|---|---|
| 芯片 | 只支持 **Ascend950**（`-s Ascend950`）|
| 卡号 | **单卡、且不能改可见卡号** → harness 里 **不要** 设 `ASCEND_RT_VISIBLE_DEVICES`；脚本里 `unset` 掉 |
| 仿真范围 | **整个应用都被仿真**，不是只仿真你指定的算子；host 侧每次 NPU 调用（含 H2D、`zeros` 等）都会被模拟 → 慢 |
| shape | 逐指令仿真，**shape 必须小**：本算子建议 `T=64 / HV=1`（≈2.5 分钟），最多 `T≤128`；不要让 app 里有 warmup 循环 |
| 权限 | root 能用但会告警；同一份输出目录要可写 |

## 2. 环境准备（247 容器内）

```bash
# 容器（CANN 9.1.0 环境入口已自动 source）
docker exec -it admin123-ppfm-test bash

# 2.1 用 wheel 里的算子（不必 pip install，解包 + PYTHONPATH 即可）
W=/data/admin123/ppfm-sim
mkdir -p $W/wheel && cd $W/wheel
python3 -m zipfile -e $(ls -t /data/admin123/flash-linear-attention-npu/dist/*.whl | head -1) .
P=$W/wheel/flash_linear_attention_npu-26.7.0.dev0.data/purelib

# 2.2 ⚠ PYTHONPATH 只能"追加"：cannsim 自己也是 python 工具，覆盖掉会 ModuleNotFoundError
export PYTHONPATH=$P:${PYTHONPATH:-}

# 2.3 harness 里必须 `import fla_npu_opp_env` 且**早于 torch_npu**
#     （它把 wheel 内嵌 OPP 写进 ASCEND_CUSTOM_OPP_PATH，CANN 初始化后再设就来不及）
unset ASCEND_RT_VISIBLE_DEVICES
```

## 3. 采集（主路径：cannsim）

harness = `sim_ppfm.py`（单次调用、无 warmup、CPU 造数、只 device 0）：

```bash
cd /data/admin123/ppfm-sim
export PYTHONPATH=$P:${PYTHONPATH:-}
PPFM_SIM_T=64 PPFM_SIM_HV=1 PPFM_SIM_VARIANT=gdn \
  cannsim record "python3 /data/admin123/ppfm-sim/sim_ppfm.py" \
  -s Ascend950 --gen-report -o /data/admin123/ppfm-sim/out_ppfm_T64_HV1_gdn
```

要点：
* `user_app` 必须是**一个字符串**（`"python3 xxx.py"`）；拆成两个参数会报 `unrecognized arguments`。
* `--gen-report` 才会生成流水；不加只有 `record/`。
* 想换 shape 直接改 `PPFM_SIM_T/HV/VARIANT`（KDA 走 `variant=kda`）。

## 4. 产出结构与"哪个是我们的 kernel"

```
out_ppfm_T64_HV1_gdn/npusim_<ts>_sim_ppfm.py/
├── npusim.log                 # 应用 stdout（能看到 harness 的 [sim]/DONE 与仿真进度）
├── record/                    # instr.bin 等原始数据
└── report/results/
    ├── kernel_<n>_reports/core_0/trace_core0.json[.gz]   ← 每个 kernel launch 一份流水
    ├── kernel_launch_<n>.db
    ├── perf_log.dump
    └── runner.py.log
```

**实测（T=64/HV=1/GDN）**：检测到 **3 个 kernel launch**，其中
* `kernel_0_reports` = **我们的 kernel**（78k 事件，含 `AIC` + `AIV0` + `AIV1` 三类核）
* `kernel_1/2_reports` = 辅助 kernel（只有 `AIV0/AIV1`，2k~3k 事件）——**看错会得出错误结论**

判断方法：看 trace 里有没有 `AIC` process（`AIV*_SCALAR`、`AIV*_VECTOR`、`AIC_CUBE`…）。

## 5. 看流水 & 快速统计

**图形化**：把 `trace_core0.json` 拖进 `chrome://tracing` 或 https://ui.perfetto.dev
（W/S 缩放、A/D 平移；字段含义：VECTOR / SCALAR / Cube / MTE1 / MTE2 / MTE3 / FIXP / FLOWCTRL / ICACHELOAD）。

**命令行快速统计**（本目录 `parse_ppfm_trace.py`，按核×流水汇总 busy 占比 + top 指令）：

```bash
python3 /data/admin123/ppfm-sim/parse_ppfm_trace.py \
    out_ppfm_T64_HV1_gdn/npusim_*/report/results
```

实测输出（三种配置都跑过；`kernel_0` 都是我们的算子）：

```
# T=128 / HV=2 / GDN（2 chunk × 2 head，更有代表性）  108848 事件，跨度 174.0
[AIC ] SCALAR:85.7%  MTE2:6.1%  FIXP:3.6%  FC:2.0%  CUBE:1.4%  MTE1:1.2%
[AIV0] SCALAR:48.1%  MTE3:16.0%  MTE2:7.6%  RVECEX:7.2%  RVECST:6.9%  RVECLD:6.8%  PUSHQ:6.8%
[AIV1] SCALAR:47.2%  MTE3:17.6%  MTE2:7.5%  RVECEX:7.2%  RVECST:6.8%  RVECLD:6.8%  PUSHQ:6.5%

# T=64 / HV=1 / GDN（1 chunk × 1 head）  78084 事件，跨度 140.4
[AIC ] SCALAR:88.6%  MTE2:5.1%  FIXP:2.6%  FC:1.7%  CUBE:1.1%  MTE1:0.9%
[AIV0] SCALAR:50.0%  MTE3:19.4%  MTE2:7.5%  PUSHQ:6.5%  RVECEX:5.6%
[AIV1] SCALAR:49.0%  MTE3:21.3%  MTE2:7.4%  PUSHQ:6.0%  RVECEX:5.5%

# T=64 / HV=1 / KDA（gk 路径）  81062 事件，跨度 149.3
[AIC ] SCALAR:88.6%  MTE2:5.0%  FIXP:2.6%  FC:1.8%  CUBE:1.1%  MTE1:0.9%
[AIV0] SCALAR:51.5%  MTE3:18.8%  MTE2:7.3%  PUSHQ:7.2%  RVECEX:5.3%  RVECST:5.0%  RVECLD:4.7%
[AIV1] SCALAR:50.2%  MTE3:20.8%  MTE2:7.3%  PUSHQ:6.8%  RVECEX:5.2%  RVECST:4.9%  RVECLD:4.6%
```

读法（与上板 msprof 的结论互相印证）：
* **AIV 侧 MTE3（UB→GM 写回）占 ~20%**，是向量侧最大单项 → 和上板 `aiv_mte3` 偏大一致；
* AIV 的 `SCALAR` 占比高是因为 1 chunk 太小（固定开销主导）；要评估计算/搬运比，请用稍大 shape（T=128）复采；
* AIC 侧 `CUBE` 只有 1% —— 极小 shape 下 cube 几乎空闲，**不能用这条数据判断算力瓶颈**。

## 6. `msprof op simulator` 为什么没用（实测记录）

同一容器里按官方/技能文档操作（`LD_LIBRARY_PATH` 指向 `.../simulator/Ascend950PR_9581/lib`，试过
带/不带 `LD_PRELOAD=libruntime_camodel.so:libnpu_drv_camodel.so`、带/不带驱动、带/不带 `--privileged`），
子进程一律在 `torch.npu.set_device(0)` 处失败：

```
RuntimeError: ... aclInit, error code is 507000
E19999: Call rtRegTaskFailCallbackByModule("GeErrorTracking", ...) fail
       Assert ((RegErrorTrackingCallBack()) == ge::SUCCESS) failed [ge_executor.cc:322]
       call acl_model init callback failed
```

连"最小 torch 程序"（`torch.ones(8,8).npu()`）也一样失败 → **不是我们算子的原因**，是该 x86 CANN 9.1.0 上
msopprof 仿真运行时初始化不通。若必须要 `visualize_data.bin`（MindStudio Insight 的代码热点图），
按 CATLASS 文档在 **aarch64** 上试：`bash build.sh --simulator …` + `LD_PRELOAD` 两个 camodel 库
（`…/tools/simulator/<SOC>/lib/libruntime_camodel.so`、`libnpu_drv_camodel.so`）——本机未验证。

## 7. 成本参考（247，容器内）

| shape | 事件数(我们的 kernel) | 墙钟（`run time`） | 备注 |
|---|---|---|---|
| `T=64 / HV=1 / GDN` | 78 k | **146.9 s**（首次，含冷启动） | 推荐起点 |
| `T=128 / HV=2 / GDN` | 109 k | **78.2 s** | 第二次起明显更快 |
| `T=64 / HV=1 / KDA` | 81 k | **61.6 s** | gk 路径同样可采 |
| `T≥256` | — | 不建议 | 事件数与墙钟都随 shape 线性涨，容易卡死 |

> 首次运行偏慢（op 包加载 / 缓存冷启动，~2.5 分钟）；同一容器里第二次起同样配置 ~1 分钟。

> 时间主要花在 **host 侧**（应用里的每个 NPU 调用都被模拟），所以 harness 要"能少一次就少一次"：
> CPU 造数 → 一次 H2D → 一次算子调用 → 一次 `synchronize` → `print("DONE")`。

## 8. 交付：给别人复现的最小清单

1. `sim_ppfm.py`（harness：小 shape、单次调用、`import fla_npu_opp_env` 在最前）
2. `run_sim_ppfm_cannsim.sh`（一条命令跑完 T/HV/variant 变体）
3. `parse_ppfm_trace.py`（trace → 核×流水占比 + top 指令）
4. 输出目录：`/data/admin123/ppfm-sim/out_ppfm_T64_HV1_gdn/`（含可复现的 `trace_core0.json.gz`）


---

## 9. `-g` 源码行号：怎么开、看到什么（2026-09-29 实测）

### 9.1 打开方式（默认关闭，产物不受影响）

kernel 的 `-g` 由 `op_host/CMakeLists.txt` 里的 **env 开关**控制：

```bash
# 单算子 OPP（~40 s）：注意 PPFM_KERNEL_G 要在 bash build.sh 之前
PPFM_KERNEL_G=1 TAG=r22dbg bash scripts/gates/exp_switch.sh ascend950 <repo>
```

* 关闭时（默认）：kernel `.o` 与生产构建**逐字节相同**（实测 `md5 275dfeac…`、169032 B）；
* 打开时：`.o` 变成 3.8 MB 且含 `.debug_line`（`readelf -S | grep debug_line` 有 3 段）。

然后照本文的仿真命令采一次（`parsim=1`，T=512/HV=8 ≈ 6 min）。

### 9.2 多了哪些产物

| 文件 | 内容 |
| --- | --- |
| `core*.veccore0_code_exe.csv` | **按源码行**聚合：`code,call_count,cycles,running_time(us)`；关 `-g` 时只有表头（40 B） |
| `core*.veccore0_instr_exe.csv` | 逐 PC：指令/addr/pipe/次数/cycles（关 `-g` 也有，但没有行号） |

想看"某个 PC 是哪一行"，用 `addr2line -e <kernel.o> <addr - load_base>`；
load base 由脚本猜（本机 = `0x10d14000`）。仓库里两个现成脚本：
`work/remote/line_attr2.py`（按文件/行聚合某类 pipe）、`work/remote/map_scalar_pcs.py`。

### 9.3 采到的事实（T=512/HV=8，core0.veccore0）

| 观察 | 数字 |
| --- | --- |
| 动态指令总数（trace 事件数） | ~7.3 k/chunk/子核，其中 **scalar+scalarldst ≈ 3.0 k（41%）** |
| 标量指令的源码归属 | **99% 落在 `*_kernel.cpp`（编译器生成的 wrapper），不是我们的 .cpp 行** |
| 向量寄存器操作单价 | ~8.5 cycle/op（`Muls`/`Cast`/`VLD`/`VST` 都一样） |
| 按阶段的 cycles（code_exe，含被调者、有重叠） | `ProcessChain` 630 k / `vec.Run` 643 k / `ApplyStateUpdates` 153 k / `StageChunk(c+1)` 120 k / `UpdateVNew` 84 k / `StageChunk(c0)` 70 k |
| 按实现的 cycles | `DataCopy` 机制（intf+impl）**839 k（13%）**、`Muls` 500 k、`Cast` 451 k、`reg_compute_datacopy` 294 k |

**结论（指导后续优化）**：AIV 的标量开销不是"我们写了太多标量代码"，而是
**每个向量/搬运内建调用展开出来的地址计算与循环外壳**；要压它，方向是
"更少、更大、依赖更松的内建调用"，不是微调某几行 C++。
(§39 的 R22b 正是这条：只把标量读提前，就拿到 −8.9%。)
