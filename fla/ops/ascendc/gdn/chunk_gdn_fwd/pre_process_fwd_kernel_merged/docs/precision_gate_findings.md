
## 精度门禁升级与"假绿"排查（2026-09-28）

背景：原 smoke 只卡 `matched`、每轮 1 次、形状窄（HV≤4、T 恒 256），且迭代脚本把 `matched/max_abs`
grep 掉 ⇒ 对竞态与窄形状会假绿。按 `outputs/PIPELINE_PARALLEL_AND_PRECISION_GATE.md` §6.1 升级：

| 层 | 内容 | 位置 |
|---|---|---|
| L0 | 静态门禁：HardEvent `Set/Wait` 配对、`PIPE_ALL`/`TQue` 计数、`PPFM_DIAG` 必须为 0 | `scripts/gates/gate_l0.py` |
| L1 | 位级回归：固定 5 用例 dump `hm`，两次构建要求逐元素完全一致 | `scripts/gates/dump_hm.py` + `cmp_hm.py` |
| L2 | smoke 判据加入 **`max_abs <= 0.05`（整体 + h/m 分半）**；补 `T=68`（尾块 4 行）/`T=1023`（尾块 63 行）/`HV=8`（GDN+KDA） | `scripts/npu_smoke_ppfm.py` |
| L3 | 竞态探针：同用例复跑 N 次 / **按用例序列**复跑 N 次，逐次与标杆比并记录 `max_abs` 分布 | `scripts/gates/race_probe.py`、`seq_probe.py` |

### 实测结论（241 device 7）

1. **L0**：`SetFlag/WaitFlag<HardEvent>` = 8/8 配对，`PPFM_DIAG=0`，`PIPE_ALL=65`（ITER7 后基线）⇒ PASS。
2. **L1**（iter7 vs iter6a 基线，各跑一次 5 用例 dump）：
   `gdn-t256 / kda-t256 / gdn-hv8 / gdn-sub` **逐位一致**（`max|diff|=0`）；
   **`gdn-t1023` 整个 head 的 h+m 不同**（32768 元素，`max|diff|=1.351`）。
3. **定位**：对**同一构建**再做 a↔b↔c 三次 dump，`a↔b` 逐位一致、**`a↔c` 在 `gdn-t1023` 差 1.182**
   ⇒ **同一二进制自身不稳定**，是真竞态；而**同一用例单独复跑 10 次全部一致**
   （`race_probe` T=1023 seed=2008：10/10 通过，`max_abs` 恒为 4.175e-03；T=256 seed=1000：10/10，
   `h_half max_abs` 恒 9.537e-07）⇒ 该竞态**与"本进程此前跑过的用例"相关**（顺序/时序相关），
   与交接文档 §5.1 对 `PPFM-37` 的观察一致（"单独跑是 PASS，全量里偶发"）。
4. **L2**：升级版 smoke（10 形状）连跑 3 轮 **30/30 PASS**（含新加的 `T=68`/`T=1023`/`HV=8` 形状，
   且 `max_abs` 全部远小于 0.05）⇒ 现门禁能挡住"少数元素错"的假绿，但**单轮仍抓不到上面那个顺序相关竞态**。

### 结论与后续

* **现门禁不再假绿**：`max_abs` 进判据 + `HV=8`/尾块形状覆盖后，任何"整 head 错"都会立刻变红。
* **竞态仍未修**：它出现在 `T=1023` 这类"非整 chunk 尾部 + 前面跑过别的形状"的组合上。
  结合交接文档 §5.3-4/5 的分析（AIC 的 C 写回与 AIV 读取之间的可见性窗口），
  **结构性解法就是计划里的 A2/P4：mm1/mm3（及后续 mm2/mm4）的 C 落点改 UB，去掉 C 的 GM 往返**。
  在 A2 落地前，`seq_probe.py`（序列复跑）应作为 L3 门禁随每轮改动一起跑。
