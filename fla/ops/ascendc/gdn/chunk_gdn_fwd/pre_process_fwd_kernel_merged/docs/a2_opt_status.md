
## A2 优化方案（mm1 的 C 直落 UB）进展与结论（2026-09-28 夜）

目标：把 mm1 的 C 从「fixpipe→GM→AIV 回读」改成「fixpipe→UB」，同时**去掉 C 的 HBM 往返**与
**跨核可见性窗口**（后者正是 `gdn-t1023` 竞态的根因）。UB 预算：复用 `UB_EXT_F` 区（32 KB，
正好是 64×128 fp32），**不额外占用 UB**；段分配改为「连续半区」以对齐 fixpipe 的 SPLIT_M。

### 尝试记录

| 版本 | 做法 | 结果 |
|---|---|---|
| ITER8（`cb9b926`/`e9b8ab8`） | fixpipe `SPLIT_M` 落 `UB_EXT_F`；AIV 从 `extBlkF_`（= `UB_EXT_F + subIdx_*16KB`）读 | **h 半边全崩**（matched≈0.01，max 155~1e36/nan），m 半边完美（1.000000/2.8e-5）⇒ C 的读取位置不对 |
| ITER10（`17eab76`） | 按"UB 按子核分 bank、两半写**同一偏移**"改：AIV 新增共享基址视图 `vTmpUb_ = ubBuf_[UB_EXT_F]`，从 `vTmpUb_[lo*CV_V]` 读 | 已提交，**待上机验证**（本轮 241 设备全部转 Critical、247 驱动未加载，无法复测） |

### 定位依据

* 失败特征**只在 h 半边**（vTmp→v_new→dH→h），m 半边（T1→T2→m）完全正常 ⇒ 问题出在 vTmp 的读取，
  不是 mmad/落点整体错。
* 仓内两个已量产算子（`chunk_fwd_h_cube.h`、`chunk_kda_fwd_fwd_h.h`）都用
  `PackedTileCopyTlaToUB<..., SPLIT_M>` + 2 参 `copyL0CToDst(tensorUb, tensorL0C)`，
  且**消费侧两个子核读同一个 UB 偏移**、各自按 `subBlockIdx*半宽` 取自己的行
  ⇒ 「两半写同一偏移（各自 bank）」是既有约定；我们原来用 `subIdx_*16KB` 偏移读就是错的。
* 两者都不调用 `SetMMLayoutTransform` ⇒ 该假设排除。

### 待做（需要一台可用的 950）

1. ITER10 上机：smoke（10 形状）→ 期望 h/m 都正常；
2. 若通过：跑 L1（位级 vs ITER7 基线，预期 h 半边会**因舍入点变化而不同**，需给解释）+ L3
   （6 独立进程序列探针，期望竞态消失，0/6 失败）+ 采集成本模型（期望 b 再降；
   同时 AIV 少一次 GM 回读、少一次 DCE/DCCI、AIC 少一次 fixpipe→GM 写）；
3. 若仍是 h 半边错：按同一套探针直接 dump「AIC 构造的 UB 张量地址 / AIV 的 `extBlkF_` 与
   `vTmpUb_` 地址 + 若干位置的值」到 `hm` 的 m 半边第 0 行（诊断用，比对时排除），
   一次把 UB 地址/分裂语义定死。

### 环境阻塞（2026-09-28 21:0x）

| 机器 | 状态 |
|---|---|
| 247 | ssh 通，但**NPU 驱动未加载**：`/dev/davinci*` 不存在、`npu-smi` 不可用（主机 17:03 重启后跑在 `6.8.0-139-generic`，`/usr/local/Ascend/driver` 下无对应 .ko）。需要 root：`sudo /usr/local/Ascend/driver/script/run_driver_ko_rebuild.sh` + `run_driver_install.sh`（或重启一次） |
| 241 | ssh 通，但 **950 设备全部 Critical**（2~7 全 Critical，只有 0 是 Alarm，均不可用）⇒ 需要宿主侧复位/重启 |
| 221 / 234 | 可达，但都是 **910B3**（A2 平台），跑不了 950 的 `CATLASS_ARCH==3510` 通路 |
