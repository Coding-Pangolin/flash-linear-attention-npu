# Ascend950（A5）测试指南 — ChunkKdaBwdWyDqkgFused

本算子在 `__CCE_AICORE__ == 310` 时走 `op_kernel/arch35/`（MicroAPI **regbase** vector + 共用 Cube/tiling）。  
本文给出 **950 板端** 的编译、精度、性能步骤；910B 回归可作对照。

## 0. 环境前提

| 项 | 要求 |
|----|------|
| SOC | Ascend950 / A5 实机（`npu-smi info` 可见） |
| CANN | 与仓约定一致的 9.x（例：`cann-9.1.0-beta.1`） |
| Python | conda 环境含 `torch` / `torch_npu` / `fla` golden 依赖 |
| triton-ascend | `FLA_NPU_SOC=ascend950` 时需满足 `setup.py` 的 A5 下限（勿用过旧 3.2.0  alone 若门禁拒绝） |

```bash
conda activate <your_env>          # 例：fzy_atk
source <CANN>/set_env.sh           # 例：.../cann/ascend-toolkit/set_env.sh 或 cann-9.x/set_env.sh
cd <repo>/flash-linear-attention-npu

# 选空闲卡：取空闲设备中号码最大者
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<id>
export ASCEND_DEVICE_ID=0          # 可见设备重映射后用 0

# 避免脏 OPP / 半成品路径
unset ASCEND_CUSTOM_OPP_PATH
unset FLA_WY_DQKG_STAGE FLA_WY_DQKG_TASK_BEGIN FLA_WY_DQKG_TASK_END
```

**重要**：先 `import fla_npu`，再 `import torch_npu`（tiling SO / 561103）。测脚本已按此顺序。

---

## 1. 编译安装（950 wheel）

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd_wy_dqkg_fused \
  python -m pip wheel --no-build-isolation --no-deps . -w dist_950

pip install --force-reinstall --no-deps --no-cache-dir \
  dist_950/flash_linear_attention_npu-*-950.*.whl
```

验收：wheel 内含

- `.../op_kernel/.../arch35/chunk_kda_bwd_wy_dqkg_fused_{regbase,vector}.h`
- `.../kernel/ascend950/chunk_kda_bwd_wy_dqkg_fused/*.o`

（可选）910B 对照编包装在 `dist/`，`FLA_NPU_SOC=ascend910b`，确认 910B 路径未误用 arch35。

---

## 2. 精度测试

### 2.1 全量 suite（必跑）

```bash
python torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_wy_dqkg_fused.py
```

覆盖（脚本内）：定长 / `state_v_first` / varlen / `split_stages` 等；期望末行：

```text
all cases passed
```

单测入口（同仓 golden）：

```bash
python fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/test/test_chunk_kda_bwd_wy_dqkg_fused.py
```

### 2.2 判据与排障

- Golden：与 Triton `chunk_kda_bwd_kernel_wy_dqkg_fused` cube-faithful（DESIGN）；**禁止**收窄 range / 无依据放宽阈值。
- 失败时：先确认装的是 **950** wheel（非 910b），再关 `split_stages`、缩小 T 二分 Stage。
- 若仅 910B 机：精度测的是父目录 vector，**不能**代替 950 regbase 精度验收。

---

## 3. 性能测试

### 3.1 模型 case（主指标）

Shape（DESIGN）：`B=1, H=HV=32, T=8192, K=128, V=128, BT=64, bf16`，`state_v_first=false`。

```bash
OUT=fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/results/prof_a5_board
mkdir -p "$OUT"

msprof op --kernel-name=ChunkKdaBwdWyDqkgFused \
  --aic-metrics=PipeUtilization,BasicInfo \
  --output="$OUT" \
  --application="python torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_model.py"
```

读数：

- 日志中的 `Task Duration(us)`（MIX_AIC），或 `OpBasicInfo.csv` 同字段。
- Host 打印的 `avg=... ms` 仅作旁证（含 launch/同步），**不以 host avg 为门禁**。

910B 近期对照（regbase 落地后父路径）：约 **3793 µs**（vs G0 **3789 µs**，无劣化）。950 板端数字另记，勿直接与 910B 比绝对值。

### 3.2 短序仿真（可选，调度诊断）

无板或需看 WAIT/BAR 时（SOC 名按本机改）：

```bash
export FLA_SIM_T=1024 FLA_SIM_H=2
msprof op simulator --kernel-name=ChunkKdaBwdWyDqkgFused \
  --soc-version=Ascend950 \
  --output=fla/ops/ascendc/kda/chunk_kda_bwd_wy_dqkg_fused/results/prof_sim_a5 \
  --application="python torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_sim.py"
```

---

## 4. 建议验收清单

| # | 项 | 通过标准 |
|---|----|----------|
| 1 | `FLA_NPU_SOC=ascend950` 编包 | 成功；含 arch35 + ascend950 `.o` |
| 2 | 精度 suite | `all cases passed` |
| 3 | 模型 msprof | 能采到 `ChunkKdaBwdWyDqkgFused` Task Duration；记录 µs 到 `ITER_LOG` / 本地 summary |
| 4 | （可选）910B 对照 | suite 绿 + Task Dur 相对 G0 无显著劣化（噪声约 ±50 µs） |

---

## 5. 相关路径

| 路径 | 说明 |
|------|------|
| [`op_kernel/arch35/`](op_kernel/arch35/) | A5 dual-path / regbase |
| [`DESIGN.md`](DESIGN.md) §7 | SOC / arch35 契约 |
| [`ITER_LOG.md`](ITER_LOG.md) | 910B 性能年表 |
| `torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_wy_dqkg_fused.py` | 精度 |
| `torch_custom/fla_npu/test/prof_chunk_kda_bwd_wy_dqkg_fused_model.py` | 模型性能 smoke |
