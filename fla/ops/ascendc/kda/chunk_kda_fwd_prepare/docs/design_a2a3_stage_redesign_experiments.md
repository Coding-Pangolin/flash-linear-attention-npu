# ChunkKdaFwdPrepare（A2/A3）Stage 重新审视：结论与实测方案

> 前提变化（2026-10-08）：**不再要求与 A5 同名同序**。原实现只是当前基线，A2/A3 可以按自身
> UB/L1 预算与同步语义**独立重切/重排**，只要满足培训 19 条约束 + 逐 Stage 可验证。
> 依据：`docs/design.md`、`op_kernel/{arch22,arch35}`、`design_a2a3_budget_4head_vs_2head.md`、
> 培训文档《Stage 划分与 4-Head 调度》。

## 1. 重新审视结论：边界不动，三处结构要重设计

| 项 | 结论 | 依据 |
|---|---|---|
| **Stage 边界（8 个）** | **不动** | 纯数学依赖 + 约束 1/2 唯一确定：`C4/C5` 不能合并（C5 依赖 C4 的 T，合并会让 cube stage 依赖本 stage 新产物）；`V0/V1` 合并无收益（内存代价相同）；`V6` 不能并入 V0/V1（各自一轮算不完）。 |
| **R1 跨核握手粒度 pair(2 head) → 4-head 组** | **要改，待实测** | A2/A3 是 mode 0x2 **集合同步**（AIC 一次 set 唤醒两个 AIV；两个 AIV 都 set 才唤醒 AIC）。现在按 pair 分两次 wait/set 是在集合语义上**人为分两批**，每 4-head 组每 stage 2 次握手；改成每组 1 次可把跨核回合 **6 → 3**。代价：AIC 要等两个 AIV 都完成才启动，失去"AIV 做 pair1 时 AIC 做 pair0"的重叠。**是省握手还是丢重叠，只能实测。** |
| **R2 拆 V3：V3a(mask+beta) / V3b(叶子逆+跨叶修正)** | **要改，待实测** | `mask+beta` 可**逐 band 早做**（C2 每完成一个 band 就有该 band 的 raw score）；只有叶子逆必须等完整 `L`（64×64 下三角）。拆开后 V3a 能与 C2 的后续 band 部分重叠，缩短 `C2→V3→C4/C5→C7` 关键路径。A2/A3 无 VF ⇒ 拆分只受 UB 约束（约束 5 允许 vector→vector 中间量留 UB 双块）。 |
| **R3 V6 提前 + 回合 3→2**（原 D1/D2） | 保持候选 | V6 只依赖 V0；与 VF 无关。 |
| **R4 C2 内部 band 流水 + C7 RHS 预取**（原 D4） | 实现级，不改边界 | C2 的 4 个 band 顺序提交，可让 band s+1 的 GM→L1→L0 与 band s 的 MMAD 重叠（L0A/L0B 有 ping/pong 空间）；C7 的 RHS 可用 L1 余量 144 KiB 预取。 |
| 不建议 | 合并 V0/V1（无收益）、合并 C4/C5（违反约束 2）、4-head 驻留 UB（需 328 KiB > 184） | 字节账见 `design_a2a3_budget_4head_vs_2head.md` |

**R1/R2 才是"A2/A3 性能不好"最可能的两个结构性原因**（前者是握手次数，后者是关键路径上的 vector 段）；
`R4` 是同一批里的低成本收益。

## 2. 234 上的实测方案（一次性会话可做完）

### 2.0 环境准备（隔离，避免影响他人）

```bash
# 1) 隔离工作目录（不碰别人的 venv / 构建树）
mkdir -p ~/ppfm-a2 && cd ~/ppfm-a2
cp -a <已有 A2 venv> ./env            # 或复用现成 venv，但 OPP 安装目录必须独立
git clone <仓> repo && cd repo && git checkout <待测 commit>
# 2) 单算子 OPP 包（A2 用 ascend910b）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=chunk_kda_fwd_prepare
# 3) 装到独立目录并切软链（每轮一个 TAG，便于 A/B 与回退）
./build_out/*.run --quiet --install-path=<pkg>/opp_vendors/<TAG>
```

### E1 整核画像（先做，回答"瓶颈在 AIV / AIC / 握手"）

```bash
for s in 1024:8 4096:8 11264:32; do
  T=${s%%:*}; HV=${s##*:}
  msprof op --application="python3 <op>/scripts/gates/perf_harness.py" \
    --output=./perf/E1_${T}_${HV} --kernel-name=ChunkKdaFwdPrepare \
    --launch-count=1 --warm-up=1 --aic-metrics=Default
done
# 变体维度：VARIANT=gdn / kda 各一组（KDA 的 V0 含逐 k gate，量级不同）
```

**判据**：读 `OpBasicInfo.csv` 的 `Task Duration` + pipe busy。
AIV(AIV busy 低) 且 AIC(cube 低) 同时低而总时长高 ⇒ **握手/等待主导 ⇒ 优先 R1**；
AIV 饱和 ⇒ **优先 R2**（V3 属 vector）；AIC 饱和 ⇒ **优先 R4**。

### E2 分 Stage 屏蔽差分（回答"哪个 Stage 占关键路径"）

对 7 个 stage 各做一版"**body 置空、同步回合不变**"的实验核，同口径测 3×10 取中位：
`V0+V1 / C2 / V3 / C4 / C5 / V6 / C7`。

- 差值 `T_k − T_all` ≈ 该 Stage 的**关键路径贡献**；接近 0 说明它已被隐藏（这本身是有用结论）。
- ⚠ 置空版**不跑精度**（数值必错），只用于时长归因。
- 只改计算、不改 flag 协议 ⇒ 不引入 hang 风险。

**判据**：贡献最大的 Stage 就是 R2/R4 的目标；若 `V3` 与 `C2` 之和 ≥ 60% 总时长 ⇒ R2 优先级最高。

### E3 R1 的 A/B（握手粒度）

| 变体 | 改法 |
|---|---|
| A（现状） | 每 pair 一次 `wait ready` / `set free`（每 stage 2 次） |
| B（R1） | 每 **4-head 组**一次：AIC 等"两个 pair 都 ready"后依次做 2 个 pair 的 C2/C4/C7；AIV 侧同理每组 1 次 `wait free`/`set ready` |

只改 flag 粒度与循环边界，**不改任何计算**。位级必须与 A 一致。

**判据**：位级一致 + 时长中位更低者胜；差异 < 2%（噪声）则维持现状。

### E4 R2 的 A/B（拆 V3）

| 变体 | 改法 |
|---|---|
| A（现状） | `V3` 一次完成 mask+beta+叶子逆+跨叶修正 |
| B（R2） | `V3a = mask+beta`（可逐 band 触发）→ `V3b = 叶子逆 + 跨叶修正`；`V3a` 在 C2 的 band s 完成后即可进行 |

实现提示：`V3a` 逐 band 可做，但需要新增"band 就绪"边（C2 band s → AIV），或者让 AIV 在 `C2` 期间
按 band 轮询（用 4 个 band 的 ready 粒度）。属于 flag 协议改动 ⇒ 必须跑 ≥3 轮 hang 看护。

**判据**：位级一致 + 关键路径（E2 的 C2+V3 之和）下降。

### E5 R4（实现级，风险最低）

- C2：band s+1 的 GM→L1→L0A/L0B 与 band s 的 MMAD 重叠（L0A/L0B ping/pong 取反 + L1 槽 wait/释放）；
- C7：在 C4/C5 期间把 `K_beta_g/V_beta` 预取到 L1。

判据：AIC busy 或 C2/C7 时长下降，位级一致。

### 共同判据

```text
功能：位级一致（ATK 200 条里挑 dense/GVA/varlen/尾块 各 ≥2 条 + TilingKey 抽样）→ 全绿后再跑 full
性能：同机同频、msprof kernel duration、3 warmup×10 取中位；改善需 ≥2~3% 且大于重复采样极差
稳定：R1/R2/E4 涉及 flag ⇒ ≥3 轮无 hang；每次 set 都有配对 wait
```

## 3. 执行顺序建议（先测后改）

```
E0 环境 → E1 画像 → E2 分 Stage 计时
   → 若"握手主导" ⇒ E3（R1）
   → 若"V3/C2 主导" ⇒ E4（R2）
   → 无论哪条，E5（R4）都可先做，成本最低
```

## 4. 当前阻塞

2026-10-08 实测：从本地与 241 都**无法访问 234 / 221**（`ping` / `port22` 均不通，只有 246/247 可达）。
⇒ E1–E5 需要其一：(a) 234 恢复可达并给出登录方式；(b) 在 241/246 上提供一个 A2(910B) 容器；
(c) 按本文 §2 命令在 234 上自行执行，把 `OpBasicInfo.csv` 与时长回传。

## 5. E1 实测结果（2026-10-08，234 / 910B3 / device 7 / 1800 MHz）

环境：自建隔离容器 `kda_a2`（镜像与 wym 同源，Ubuntu22.04 + CANN 9.2.0-beta.1），
宿主 conda py3.10(`fzy`) + torch 2.12 + torch_npu 2.12，`PYTHONPATH` 指向 `/home/admin123/kda_a2/python-a2c`
（fla_npu + OPP 副本），直调稳定入口（`A_log=` 大写；executor 里写的 `a_log=` 与该 ABI 不符，已绕开）。

### 5.1 端到端（`torch.npu.Event`，warmup=2 repeat=5）

| 用例 | shape | 时长 |
|---|---|---|
| `case 63` | B1 HK96 HV96 T8192，BNSD，forward | **18,850 µs/call** |

对照：该 shape 的 **A5 目标是 2700 µs** ⇒ A2 现状约 **7×** A5 目标。
（`msprof op` 同用例 Task Duration = 18,812 µs，与 event 一致 ⇒ 口径可信。）

### 5.2 分 pipe busy（`msprof op --aic-metrics=Default`，20 AIC / 40 AIV）

| 单元 | 时间 | 占比 | 说明 |
|---|---:|---:|---|
| **AIV vec** | **12,607 µs** | **67.0%** | **主瓶颈** |
| AIV scalar | 4,726 µs | 25.1% | 其中 `aiv_scalar_stall_by_ub = 5,845 µs`、`aiv_scalar_vector_stall = 6,208 µs` |
| AIV mte2 / mte3 | 607 / 544 µs | 3.2% / 2.9% | 搬运不是问题 |
| AIC busy 合计 | ≈3,441 µs | **21%** | cube 仅 **264 µs (1.4%)**；fixpipe 823 (4.4%)、mte2 722 (3.8%)、mte1 371 (2.0%)、scalar 1,261 (6.7%) |
| AIC `scalar_wait` | 7,594 µs | — | AIC 一半时间在等跨核 flag（但它不是瓶颈） |

**E1 结论（可直接影响设计）**：

1. **A2 上该算子是 AIV-vec 受限**：vec 67% + scalar 25% ≈ 92% 占用，cube 只有 1.4%
   ⇒ 重设计的杠杆是**减少/并行 vector 工作**，不是 cube 侧、也不是搬运侧；
2. **AIC 有大量余量（79% 空闲）** ⇒ "把 elementwise 从 AIV 挪到 AIC / 让 cube 多做一点"
   这类思路在 A2 上比在 A5 上更有价值；
3. **R1（握手粒度 pair→4-head）的优先级下降**：它的收益来自"省跨核往返"，而 AIC 本来就有
   7.6 ms 在等 flag、且不是瓶颈 ⇒ 先把 vector 工作量降下来更划算；
4. **R2（拆 V3）需要 E2 才能定价**：必须知道 V3 在 12.6 ms vec 里的占比
   （若 `V3` 占大头，拆 V3 让 `V3a` 与 C2 重叠才有意义；若大头在 V0/V1/V6，则要换方向，
   例如 V0 的 norm/gate 是否可按 HK 去重、V6 的 exp/cast 能否合并）。

### 5.3 E2 进展

分 stage 屏蔽差分需要重编算子（`build.sh --pkg --soc=ascend910b`）。
首次失败原因：我 rsync 源仓时 `--exclude build` 把 `cmake/third_party/build/modules/patch/*`
也排除了（该目录由上次构建生成，含 `protobuf-hide_absl_symbols.patch` 等）⇒ 已补回并重新构建。
