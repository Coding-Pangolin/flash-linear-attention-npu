# 竞态（`gdn-t1023` h 半边整头崩）现状 · 根因 · 独立复核

> 2026-09-29，独立窗口（只读主仓 + 独立沙箱：`~/BartonFang/ppfm-race` + `~/BartonFang/envs/race`）。
> 不覆盖本文档之外的算子改动；不含任何算子源码修改建议（除 §7 两条一行级）。

## 0. TL;DR

1. **已解决，且是结构性解决**：`vTmp`（mm1 的 C）从「fixpipe→**GM**→AIV MTE2 回读」改成
   「fixpipe→**UB**」（ITER10 / `PPFM_VTMP_UB=1`，`17eab762` 引入、`2925673` 起被 §19 门禁覆盖）。
   之后 R9（`0c06582`）才能安全删掉 4 处"过渡探读"。
2. **我独立复核通过**（241，两台里当年复现过竞态的那台）：
   2 组配置 × 16 个独立进程 = **32 进程 0 失败**，其中一组是**最可疑组合**
   （把 `vTmp` 改回 GM 路径 + 探读删掉）。加上 §20 在 247 的 20 进程 0 失败 ⇒ **三组独立样本**。
3. 32 次 `gdn-t1023` 的 `h max_abs` **只有唯一取值 `1.633e-03`**（历史命中时是 0.91 / 1.23 的跳变）
   ⇒ 时序窗口消失，不是"变低保真"。
4. **残余**：`dH`/`T2`（mm2/mm4 的 C）仍是同类的跨核 GM 边（950 上无 DCCI/DSB、只有 flag）。
   未见过故障，但结构同类 ⇒ 建议按 **R6（dH/T2 走 CV）** 一并收口。
5. **不需要每轮都跑进程级 soak**；只在改动触及「同步 / 落点 / UB-L1 槽位 / flag 计数」时跑，
   并用 §5 的"标杆缓存 + 并发批次"把 16 进程从 ~56 min 压到 ~20 min。

## 1. 失败形态与回归口径（判据沿用，别改）

| 项 | 值 |
|---|---|
| 敏感用例 | `gdn-t1023`（T=1023，尾块 63 行）；历史上 `gdn-sub`（子区间 `[40,512]`）也命中过 |
| 触发条件 | **独立进程**（同一进程内复跑 8 轮 0/40 全过 ⇒ 单进程不敏感）+ 序列前跑过别的形状 |
| 失败形态 | **h 半边整 head 崩**，`max_abs` 0.83~1.4；**m 半边正常** |
| 历史概率 | 计划 §1.4 记 ~1/6；`dddecd1`（ITER7）那次 6 进程里 4 个出错；TILE=0 基线同样复现 |
| 判据 | 每用例 `matched ≥ 0.999` **且** `max_abs ≤ 0.05`（h/m 分开算）；`race_probe_procs.sh` 的总 `FAIL==0` |
| 序列 | `scripts/gates/dump_hm.py` 的 5 例：`gdn-t256 / kda-t256 / gdn-t1023 / gdn-hv8 / gdn-sub` |

统计常识：若真实概率仍是 1/6，20 个独立进程全清的概率 ≈ 2.6%；**两组样本全清 < 0.2%**。

## 2. 根因证据链

### 2.1 症状 → 唯一自洽的解释

AIV 读到 AIC 写的 `vTmp` 的**上一代 / 未写值（全 0）**
⇒ `v_new = (v − 0)·dg ≈ v` ⇒ `dH = kᵀ·v_new` 错 ⇒ `h = decay⊙h + dH` 整头错。
而 `m` 走 `T1 → T2` 另一条链，所以"h 崩、m 正常"。

当年 `PPFM_DIAG` 指纹直接抓到过这一幕：**AIC 实际写了 −0.013 / +0.0092，AIV 读到 `vTmp` 全 0**。

### 2.2 为什么"加屏障"治不好（历史记录已排除的路）

计划/validation 记录里试过且**无效**的手段：`IterateAll` 显式等待、reader 侧 DCCI、
writer 侧 DCCI + `DataSyncBarrier<DDR>`、双缓冲推迟读、`PIPE_ALL`、UB 预算调整。
ITER5（合并两次跨核通知）**零收益** ⇒ 瓶颈不是 flag 往返。

结论：**这是"中间量落点结构"问题，不是"flag 少一次 / 屏障不够"**。

### 2.3 新证据（本文档补的记录里没有的两条）

**(a) 仓内两个已量产 950 算子根本不走这条边。**

`chunk_fwd_h/op_kernel/arch35/chunk_fwd_h_cube.h`：

- C 由 `CopyL0CToUbS0` 直接 fixpipe→**UB**（带 subblock 索引 `head.aiv`），**不落 GM**；
- 跨核 wait 全排在**消费流水**上：`CrossCoreWaitFlag<0x4, PIPE_MTE2>`（后面是 GM→L1 拷贝，`:351/:508`）、
  `<0x4, PIPE_V>`（vector 消费，vec `:668/:870/:1131...`）、`<0x4, PIPE_FIX>`（FIX 槽位复用）；
- `PIPE_S` **只用于 round 级 DONE/ACK 的"下发闸门"**，源码注释原话：
  "PIPE_S 将 round 握手放到控制流水，阻止下一轮 MTE2 预取越过 DONE/ACK"、"ACK 用 PIPE_S gate 下一轮全部指令下发"。

我们原来的写法（C 走 GM + 只有 ready 边没有 free 边 + 用 `PIPE_S` 等数据）与量产惯用法两处都不同。

**(b) 那 4 处"过渡探读"在机理上是"下发栅栏"，不是可见性修复。**

`DataCopy(row2F_, vTmpF_/t1F_/dhBuf/t2Buf, 8)` 紧跟一个 `PipeBarrier<PIPE_ALL>()`，
位置就在跨核 wait 之后、真实消费之前 ⇒ 它靠**全流水排空**把窗口关掉，属于掩盖时序。
所以：**先做落点结构修复、再删探读** 的顺序是对的（ITER10 → R9）；反过来先删探读就会重新暴露。

## 3. 我的独立复核（数字）

沙箱：241 上独立 conda 环境副本 + 独立仓副本 + 独立 OPP 安装目录（`opp_vendors/*`），
只用 **device 1**（空闲），未触碰主 venv / 主仓 build 树 / 246 / 221。

| 组 | 机器 | commit | 配置 | 独立进程 | 结果 |
|---|---|---|---|---|---|
| §20 记录（主线程） | 247 | `0c06582` | `VTMP_UB=1` + 探读删（shipped） | 20 | 0 失败 `RACE_PROBE_CLEAN` |
| **A（我）** | **241** | `0c06582` | **`VTMP_UB=0`（GM 路径）+ 探读删** | **16** | **0 失败 `SOAK_CLEAN`** |
| **B（我）** | **241** | `0c06582` | `VTMP_UB=1` + 探读删（shipped） | **16** | **0 失败 `SOAK_CLEAN`** |

- 每进程 = 新进程，跑同一 5 用例序列；共 **160 次用例运行，0 失败**。
- **32 次 `gdn-t1023` 的 `h max_abs` 取值集合 = {1.633e-03}**（唯一值）。
- 副产品：A 组同时验证了"950 上 `PIPE_S` 能挡住后续下发"这个假设——
  在 GM 路径 + 没有任何探读栅栏 + 只有 `PIPE_S` 等待的条件下 16 进程全清。
  ⇒ **不建议**在没有新证据时把 950 的 `PIPE_S` 改成 `PIPE_MTE2`（那可能是减弱）。

## 4. 检测成本：为什么"每轮都跑"很贵，怎么压

241 上单进程 ~3.5 min 的**实测**构成（`race_timing.py`）：

| 阶段 | 耗时 | 占位 |
|---|---|---|
| **ACL / `torch_npu` 初始化** | **~150 s** | 大头，与算子无关 |
| CPU 标杆（5 用例，`reference/reference.py`） | ~58 s | 可缓存 |
| **kernel 5 次实跑 + 对拍** | **~1.9 s** | 真正"测算子" |

⇒ 20 进程 ≈ **50 min，其中算子的部分不到 1 min**。

两个降本手段（已实测）：**标杆缓存**（每进程省 58 s）、**并发批次**（`PAR=4`，16 进程 ~56 min → ~20 min）。

**推荐协议**

- **每轮（~1 min）**：L1 位级 + L2 smoke（含 `max_abs≤0.05`）+ 41 条用例 —— 已能挡住除本竞态外的一切；
- **触发式**：仅当改动触及 **同步 / 落点（GM↔UB↔L1）/ UB-L1 槽位 / flag 计数** 时，加跑进程级 soak，
  用 `PROCS=20 PAR=4`（脚本见 §5），**后台跑、不阻塞下一轮**；
- **判据不放松**：单进程复跑对这类竞态不敏感（历史 0/40 vs 进程级 4/6），别用"多跑几轮"代替。

## 5. 复核用脚本（本次新增，已放进 `scripts/gates/`）

| 文件 | 用途 |
|---|---|
| `race_probe_cached.py` | 序列探针（与 `seq_probe.py` 同序列、同判据），`want` 走盘上缓存 ⇒ 每进程省 ~58 s |
| `race_soak.sh` | 进程级 soak：`PROCS` / `PAR` / `DEV` / `TAG` 可控，输出每进程逐用例 h/m `max_abs` 与脏进程数 |
| `race_timing.py` | 阶段计时（ACL / 标杆 / kernel），用于判断"贵在哪" |

用法：

```bash
OP=<算子目录>
PROCS=20 PAR=4 DEV=1 TAG=r9 bash scripts/gates/race_soak.sh "$OP"
# → 末行 SOAK_CLEAN / SOAK_DIRTY；日志 /tmp/race_soak_<TAG>/proc_*.log
```

> 注意：单算子 OPP 快路径脚本 `exp_switch.sh` 在**全新**环境上会踩一个坑，见 §7.1。

## 6. 残余风险与建议

| 边 | 现状 | 风险 |
|---|---|---|
| `vTmp`（mm1 的 C）AIC→AIV | **fixpipe→UB**（`VTMP_UB=1`） | 结构上已消除 ✅ |
| `T1`（mm3 的 C） | fixpipe 直接按 bf16 落 GM，**只有 AIC 用**（mm4 的 B） | 不跨核 ✅ |
| **`dH`/`T2`（mm2/mm4 的 C）AIC→AIV** | **仍走 GM**；950 上 `LEGACY_CACHEOPS=0` ⇒ 无 DCCI/DSB，仅 flag | 与当年 `vTmp` **结构同类**；未见过故障，但未证伪 |
| AIV→AIC：`wBf_/hBf_/mBf_/kBf_/lBf_/vNewBf_` | 走 GM；同上 | 同类；目前只靠 flag |

建议：把 **R6（`dH`/`T2` 走 CV/UB）** 当作"竞态的另一半"来做——它同时是 §20 之后继续压 `b` 的正路
（模型 case 现 2.29× H20）。UB 预算不够时需要先做 UB 回收，或先只把 `dH` 走 CV（计划里已有这个折中）。

## 7. 顺手发现的两处小问题

### 7.1 `scripts/gates/exp_switch.sh` 在全新环境上会失败（一行修）

`--install-path` 的安装器**不会自建中间目录**：新机器/新 env 上 `opp_vendors/` 不存在时报
`[ops_custom] ... create <pkg>/opp_vendors/<TAG> failed`（本次在隔离 env 上实测到）。

修法（`rm -rf "$EXP"` 之前加一行）：

```bash
mkdir -p "$(dirname "$EXP")"
```

### 7.2 `docs/a2_opt_status.md` 的结论已过时

该文仍写 ITER10「已提交，**待上机验证**」、ITER8「h 半边全崩」。
实际：`2925673`（§19 R0 基线）与 `a3de9f2`（§20 probe0 测量）两处源码里
`PPFM_VTMP_UB` 在 950 分支都是 **1**，且两轮门禁全绿 + 进程级 0/20 ⇒ **已经验证可用**。
建议在该表补一行「✅ 已验证（§19/§20）」，否则下一个人会以为 UB 落点还不能用。

## 8. 沙箱位置（用完可整体删除）

- 241：`~/BartonFang/ppfm-race/`（`repo`、`runs/*.run`、`ref_cache/`、`soak_*.out`、`race_*.{sh,py}`）
  、`~/BartonFang/envs/race/`（独立 conda env 副本，2.5 G）、日志 `/tmp/race_soak_r9*`
- 本地：`work/race_sandbox/`（同一套脚本与 patch）

清理：`rm -rf ~/BartonFang/ppfm-race ~/BartonFang/envs/race`

## 9. 复现本文的完整命令（供后人一次性跑）

```bash
# 1) 隔离沙箱（不动主 env / 主仓）
cp -a ~/BartonFang/envs/fzy ~/BartonFang/envs/race
cp -a ~/BartonFang/flash-linear-attention-npu ~/BartonFang/ppfm-race/repo   # 或按 commit 同步

# 2) 单算子 OPP 包（~1 min）+ 切到沙箱 env 的 vendor 目录
cd ~/BartonFang/ppfm-race/repo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export FLA_NPU_DISABLE_PTH=1
bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=pre_process_fwd_kernel_merged
#   把 build_out/*.run 装到 <沙箱 env>/lib/.../fla_npu/opp_vendors/<TAG> 并软链 vendors 目录

# 3) 进程级 soak（20 进程 / 4 并发）
PROCS=20 PAR=4 DEV=1 TAG=verify bash <op>/scripts/gates/race_soak.sh <op>
```
