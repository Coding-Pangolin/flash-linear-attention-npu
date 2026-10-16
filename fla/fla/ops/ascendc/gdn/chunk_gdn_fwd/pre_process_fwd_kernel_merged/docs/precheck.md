# catlass_chunk_fwd_h_pre_process 平台与开发环境预检

日期：2026-09-17　设备：192.168.13.241（sz-blue-950pr-13-241）　执行账号：npu_user7

## 1. 软件与工具链

| 项目 | 取值 | 依据 |
| --- | --- | --- |
| CANN | 9.1.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后读取 `ASCEND_HOME_PATH` |
| 驱动 / `npu-smi` | 25.7.rc1.7 | `npu-smi info` |
| 编译器 | `bisheng`（`/usr/local/Ascend/cann-9.1.0/bin/bisheng`） | `command -v bisheng` |
| CMake | 3.28.3（`/usr/bin/cmake`） | `cmake --version` |
| 性能工具 | `msprof`（`/usr/local/Ascend/cann-9.1.0/bin/msprof`） | `command -v msprof` |
| Python | 3.10.20（conda env `fzy`：`/home/npu_user7/BartonFang/envs/fzy`） | `conda activate fzy && python -V` |
| PyTorch | 2.7.1+cpu，torch_npu 2.7.1.post5 | `python -c "import torch, torch_npu"` |
| NumPy | 1.26.4 | 同上 |
| CATLASS 源码 | 工作区根 `./catlass/`，含 `include/`、`examples/`（103 个子目录）、`docs/` | `verify_catlass_ready.sh` 通过（`VERIFY_EXIT=0`） |

`python -c "import torch"` 在未加载 CANN 环境脚本时会因 torch_npu 后端自动加载失败报
`RuntimeError`；加载 `set_env.sh` 后正常，且 `torch.npu.is_available() == True`。

## 2. 设备与平台能力

设备 0（本算子本轮使用）：Product Name `A310-50-C00MM304A1`，Chip Name `Ascend950PR`，
NPU Name `9579`，Chip Version V100，`npu-smi` 显示 Health OK。本机共 3 张卡（NPU 0/1/2）。

平台参数取自 `${ASCEND_HOME_PATH}/../latest/acllib/data/platform_config/Ascend950PR_9579.ini`
（该文件与 `npu-smi` 报出的 NPU Name 一致）：

| 参数 | 值 |
| --- | --- |
| `SoC_version` / `Short_SoC_version` | `Ascend950PR_9579` / `Ascend950` |
| `NpuArch` | `3510`（对应 `CATLASS_ARCH=3510`，`ascend950` 目标） |
| `AIC_version` | `AIC-C-310`（`dav-c310-cube` / `dav-c310-vec`） |
| `cube_core_cnt` / `vector_core_cnt` | 28 / 56（`cube_vector_combine=split`） |
| `l0_a_size` / `l0_b_size` / `l0_c_size` | 64 KiB / 64 KiB / 256 KiB |
| `l1_size` | 512 KiB |
| `ub_size` | 253952 B（248 KiB，单 AIV 用户可用） |
| `l2_size` | 128 MiB |
| `cube_freq` | 1650 MHz |
| `cube_m/n/k_size`、`vec_calc_size` | 16 / 16 / 16、128 |
| `ubblock_size` / `ubbank_size` / `ubbank_num` | 32 B / 4096 B / 16 |
| `support_bf16` | 1 |

## 3. 与本算子相关的结论

1. **架构号**：工程编译按 `CATLASS_ARCH=3510` 配置。
2. **核数**：Cube 28、Vector 56。本算子 `h`、`m` 两条链都需要 Cube 与 Vector 配合，
   任务切分不能按 56 个 Cube 核规划。
3. **UB 容量**：单 AIV 可用 248 KiB，`h` 的 `64 x BLOCK_SIZE` FP32 累加块（`BLOCK_SIZE=64`
   时 `64 x 64 x 4 B = 16 KiB`）与 `m` 的 `BK1 x BLOCK_SIZE` 累加块量级相同，容量需求与
   上游 `BLOCK_SIZE` 两档（32 / 64）的划分一致；具体 slot 与流水在 03 阶段按 R01–R21 计算。
4. **dtype**：平台支持 BF16，满足 `k/w/u/v/bg` 的 BF16 输入与 `exp2` 路径。
5. **精度路径**：CPU 标杆在 `fzy` 环境以 PyTorch CPU 运行，不依赖 NPU，满足标杆独立性要求。
6. **无 GPU**：本机没有 NVIDIA 设备，上游 Triton kernel 只能作为功能对照阅读，不能直接运行
   采集参考结果；标杆按本流程在 CPU 侧独立实现并交叉验证。

## 4. 待办与风险

- 本机为共享设备（账号 `npu_user7` 由多人使用），性能采集前需确认卡上无其他负载，
  否则 `Task Duration` 不可比。
- `torch_npu` 与 CANN 的 owner 校验告警（`owner does not match the current owner`）为已知现象，
  不影响功能，仅需在日志中忽略。