# pre_process_fwd_kernel_merged：910B / 910_93（Atlas A2/A3）移植报告

> 分支 `feat/ppfm-tile-a5`。950 侧行为保持不变（同一份 kernel，按 arch 分档），
> 910B 侧从"能编译"一路修到"精度与 950 同档、性能略优于 950"。
> 采集机：221 容器 `wym`（8×910B3，CANN 9.1.0）；950 对照：246 容器 `admin123-gdn-test`。

## 1. 结论

| 项 | 950（Ascend950PR） | 910B（910B3，Atlas A2） |
|---|---|---|
| 构建 | ✅ `FLA_NPU_SOC=ascend950` | ✅ `FLA_NPU_SOC=ascend910b`（128 个编译错误全部消除） |
| L0 静态门禁 | ✅ | ✅ |
| L2 smoke（10 形状，含 `max_abs<=0.05`） | 10/10 | **10/10** |
| L4 全量 41 条 | 41/41（`--repeats 2`） | **41/41** |
| L3 序列探针（独立进程复跑） | 最近一轮 0/30 | **0/20** |
| h 半边 max_abs（T=256 同用例） | 9.537e-07 | **9.537e-07**（同档） |
| m 半边 max_abs | 2.766e-05 | **2.766e-05**（同档） |
| 成本模型 `a / b`（msprof `Task Duration`） | 74.5 µs / 14.08 µs·chunk⁻¹ | **72.0 µs / 12.27 µs·chunk⁻¹** |
| 模型 case T=11264/HK=HV=32 | 5.10 ms（3.15× H20） | **4.53 ms（2.79× H20）** |

## 2. 平台分档表（都在 `op_kernel/pre_process_fwd_kernel_merged.cpp` 头部）

| 项 | 950 | 910B/910_93 | 依据 |
|---|---|---|---|
| 判别宏 | `__CCE_AICORE__ == 310` | 其余 | 同仓 `chunk_fwd_h.cpp` 的分发写法 |
| `PPFM_ARCH_IS_950` | 1 | 0 | — |
| `CATLASS_ARCH` | 3510 | 2201 | 必须在含 catlass 头之前定义 |
| gemm arch tag | `Catlass::Arch::Ascend950` | `Catlass::Arch::AtlasA2` | `PackedTileCopyTlaToUB`/`CopyL0CToUBMode` 只存在于 3510 |
| C 回写别名 | `CopyL0CToDst`（3510） | `CopyL0CToGm`（2201） | 两代 API 名字不同 |
| `PPFM_VTMP_UB` | 1（L0C→UB 直连） | **0（必须经 GM 中转）** | A2/A3 无 L0C→UB 通道 |
| 跨核 flag 模式 | `0x4`（按 subblock 分槽） | **`0x2`（AIC:2×AIV 集合同步）** | `chunk_fwd_h_policy.h` 的既有范式 |
| AIC 侧 set/wait 次数 | 每方向 2 次 | **每方向 1 次** | 0x2 下一次 set 即对本 block 两个 AIV 置起 |
| 等待排队流水 | `PIPE_S` | `PIPE_MTE2`（消费方是 MTE2） | 与 arch22 生产算子一致 |
| `PPFM_LEGACY_CACHEOPS` | 0 | 1（DSB/DCCI；实测不是本问题的根因，作为保守项保留） | — |

## 3. 移植中真正踩到的 4 个坑（按发现顺序，都带可复现证据）

### 3.1 编译期：三类 3510 独有符号 + arch tag

`PackedTileCopyTlaToUB` / `CopyL0CToUBMode` / `TiledCopyNTSplitUb` 在 CATLASS_ARCH=2201 下不存在；
`DEPENDENT_FALSE<Ascend950>` 说明 arch tag 必须是 `AtlasA2`；`CopyL0CToGm` 与 `CopyL0CToDst` 两代命名。
→ 全部按 `PPFM_ARCH_IS_950` 分档（提交 `0a9eea6`）。修完 910B 一次编译通过。

### 3.2 跨核 flag：0x2 是"AIC 与 2 个 AIV"的集合同步

按 950 的"每子核一个 slot、AIC 侧 set/wait 两次"照搬会挂死（AIC 等 `id+16` 永不被置起）。
改成 AIC 侧一对 set/wait（提交 `02db99b`）后不再挂死。

### 3.3 `m ← I` 的 prologue 在 A2 上退化（m 半边全错）

**证据（零输入探针）**：把 `w=v=u=0、g=gk=0`（此时必须 `h≡0、m≡I`）喂进去，
得到 `h=0` 正确，但 **m 每行都是常数**：`m[r][:] ≡ 1`（r%4∈{0,1}）或 `≡0`（r%4∈{2,3}），
即整条 m 链保持秩 1。用 `w=0` 单独隔离后，m 仍是"行常数" ⇒ 不是 mm 链，是初值。
**根因**：A2 上 `ArithProgression` 走 common 实现（标量写 8 拍 + 向量 `Adds` 展开），
与外层"逐行向量组合造对角"叠加后不可靠；全仓只有本算子用到该原语（无先例可对照）。
**修法**：950 保留原构造；910B 改成逐行 `Duplicate` 清零 + 单点 `SetValue`，行间 `PIPE_ALL`（提交 `a39f362`）。
**效果**：`m max_abs` 从 9.7e-01 → **5.96e-08（T=64）/ 6.99e-10（g=0）**。

### 3.4 AIC 读到的 `bf16(h)` 是脏数据（h 半边整体偏 1.5e-2）

**证据（`PPFM_RD_PROBE` 定点探针，把 AIV 侧各量搬进 hm 的 m 半边前 6 行）**：

| 探针 | 内容 | 实测 |
|---|---|---|
| row2 | prologue 写完立刻回读 fp32 `h` 状态 | **0（正确）** |
| row3 | 更新循环里读到的 fp32 `h` 状态 | **0（正确）** |
| row1 | **GM 里的 `bf16(h)` 第 0 行** | **1.2e-3 量级脏数据 ✗** |
| row4 | AIV 读到的 `dH` | 非零（≈1e-2）✗（`vTmp` 被脏 `bf16(h)` 污染所致） |

即：**fp32 状态是对的，落到 GM 的 `bf16(h)` 是错的**，AIC 的 mm1 于是把非零的 `bf16(h)`
当输入，`vTmp=W@h≠0 → dH≠0 → h 半边偏 ~1.5e-2`（m 链不受影响，所以只有 h 半边坏）。
另外用哨兵探针确认 AIC 的 C 回写本身是好的（m 链逐位精确、t1F_ 的哨兵被覆盖）。
**根因**：A2 上"**一次 `Cast` 出 bf16 行 + 循环内 128 次复用同一 UB 行走 MTE3**"的写法会写坏 bf16 落点
（同一循环里的 fp32 版本正常；m 初值用逐行写法因此一直是对的）。
**修法**：910B 改成逐行 `Duplicate → Cast → DataCopy`，行间 `PIPE_ALL`（提交 `a65df3e`）。
**效果**：`bf16(h)` 回读=0；零输入 h=**0.000e+00**；T=64 h=**3.576e-07**；T=256 h=**9.537e-07**（与 950 同档）。

> 复现探针开关（默认关，只影响诊断构建）：
> `PPFM_RD_PROBE=1`（搬 6 行探针到 hm）、`PPFM_SENTINEL_PROBE=1`（给 C 缓冲预置哨兵）。
> 判定脚本：`scripts/gates/` 之外的临时脚本见交接文档 `work/remote/dbg910b_m.py`
> （支持 `ZERO_ALL/WG_ZERO/G_ZERO/V_ZERO/SOLVE_VTMP/RUNS` 等隔离手段）。

## 4. 环境（221）两个实操坑

1. **容器盘满**：`/` 的 `statvfs.f_bavail == 0`（54.9 GB 都在 ext4 root 保留池里，root 仍能写），
   第三方 OPP `.run` 安装器据此拒绝解包（连 7 MB 都"放不下"）。
   绕过：`FLA_NPU_RUN_TMPDIR`/`TMPDIR` 指向 `/home`（`bavail>0`），
   并在 setup.py 里给安装器的临时目录加了环境变量覆盖（**仅 221 沙箱改动，未进主线**）。
2. **UB 预算**：910B/A2 每 AIV 192 KiB；本算子布局 `PPFM_VEC_UB_BYTES = 188096 B = 183.7 KiB` ✅ 放得下
   （架构上 910B 的两个 AIV 各有独立 UB，`subIdx_` 分槽只是沿用 950 的写法，不影响正确性）。

## 5. 复现命令

### 5.1 950 视角的等价性证明（不需要重建）

246 的 GitHub 拉取本轮回不上（GnuTLS），而它**入站大包会被 reset**，
所以最终提交没法直接落到 246 重编。用 `scripts/gates/arch_view_diff.py` 做等价性证明：

```bash
git show 0a9eea6:<kernel.cpp> > /tmp/old.cpp
python3 scripts/gates/arch_view_diff.py --ignore-comments \
    --macros "__CCE_AICORE__=310,PPFM_TILE_MMAD=1" /tmp/old.cpp <kernel.cpp>
```

结果：950 视角活跃代码 896 → 898 行，**唯一差异是新增的两行探针宏定义（`PPFM_SENTINEL_PROBE 0`、
`PPFM_RD_PROBE 0`），没有一行代码变化** ⇒ 4 个修复全部落在 `#else`（910B）分支或默认关的探针里，
950 的已验证二进制与最终提交行为一致。
（同一工具按 `--macros "__CCE_AICORE__=220,..."` 跑 910B 视角，则能看到上述 4 处修复的真实差异。）

```bash
# 910B 构建+安装（221 容器 wym）
cd /root/ppfm910 && source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TMPDIR=/home/barton_tmp FLA_NPU_RUN_TMPDIR=/home/barton_tmp/run-installer
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=pre_process_fwd_kernel_merged python3 scripts/build_wheel.py
python3 -m pip install --force-reinstall --no-cache-dir --no-deps dist/*.whl

# 门禁（L0 → L2 → L4 → L3）
export REPO=/root/ppfm910 STAGE=all DEV=7 ROUNDS=6
bash /tmp/gate_ppfm.sh          # 内容见交接文档 work/remote/gate_ppfm.sh

# 上板性能（msprof op）
MSOP=4050:8 ROOT=/root/ppfm910 bash run_msopprof_910b.sh 1024:8 4096:8 11264:32
```

## 6. 未完成 / 风险

* **950 的残余跨核竞态仍在**：`gdn-t1023` 在独立进程复跑中历史上有 ~1/6 概率整 head 崩
  （本次回归 0/30、0/20 未复现）；TILE=0 基线同样复现，属基线问题不是本次改造引入。
* 910B 侧本轮只做了**单卡单实例**性能采集，未做多实例并发/长稳；
  `usedAicNum = min(Nwork, aicNum)` 在 910B（20 AIC）下模型 case 仍是 2 波，链拆分（P5）收益同 950。
* `PPFM_LEGACY_CACHEOPS` 在 910B 上默认 1 是保守选择（实测开关它不影响这两处根因的真实性）；
  若要进一步压性能，可在 910B 上单独 A/B 关掉再做回归。
