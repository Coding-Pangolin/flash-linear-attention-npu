# pre_process_fwd_kernel_merged（AscendC 算子工程，WIP）

CP（context parallel）前处理算子：把一个 token 窗口压成仿射链 `(h | m)`，与竞品
`fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` 数值对齐（对标 1.0x H20）。

完整语义、Stage 划分、内存分配与验收工程见 `docs/`（`api.md` / `design.md`）与
`tests/atk/pre_process_fwd_kernel_merged/`。

## 当前进度（2026-09-28）

| 部分 | 状态 |
| --- | --- |
| 01 接口 / 02 标杆 / 03 设计 | ✅ 冻结（见 `docs/`；标杆 `reference/reference.py` 已与 H20 `ieee` 对齐） |
| Python 接入（ctypes） | ✅ 已在 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 落地（`aclnnPreProcessFwdKernelMerged` + `npu_pre_process_fwd_kernel_merged`），10 条离线单测通过 |
| op_host | 🚧 已写 `*_def.cpp` / `*_tiling.{h,cpp}` / `op_api/*`（含 aclnn 两段式）；**尚未接线进构建** |
| op_kernel | 🚧 待写（v1 计划：AIV-only 向量版先把数值打通，再换 Cube/tile 版做性能） |

## 接线方法（kernel 写完后）

1. 加 `op_host/CMakeLists.txt`（内容照 `../recompute_w_u_fwd/op_host/CMakeLists.txt`：
   `add_op_to_compiled_list()` + `target_sources(op_host_aclnnExc PRIVATE pre_process_fwd_kernel_merged_def.cpp)`
   + `add_modules_sources(OPTYPE pre_process_fwd_kernel_merged ACLNNTYPE aclnn_exclude)`
   + `add_ops_compile_options(OP_NAME PreProcessFwdKernelMerged ...)`）；
2. 加 `op_kernel/pre_process_fwd_kernel_merged.cpp`（kernel 入口，`TILING_KEY_IS(1..3)` 分派 gate 模式）
   与其实现头文件；
3. `bash build.sh` 编译，按错误逐条清；
4. 用 `reference/reference.py` 对拍精度，再与 H20 基线比性能。

**只编译本算子（快）**：`build.sh` 支持 `--ops=` 白名单，例如

```bash
bash build.sh --opkernel --soc=ascend910b --ops=pre_process_fwd_kernel_merged   # 只编 kernel
bash build.sh --ophost   --ops=pre_process_fwd_kernel_merged                    # 只编 host 侧
```

容器里的 CANN 在 `/usr/local/Ascend/cann-9.1.0`（`ASCEND_HOME_PATH` 已设），可以离线编译。

## 关键契约（细节见 `docs/api.md`）

* `B ≡ 1`（varlen 打包）；`cu_seqlens` 必给且**允许子区间**（`bos > 0` / `eos < T`），与竞品调用形态一致；
* `K = V = 128`、`chunk_size = 64`、布局 BNSD `[1, H, T, D]`；
* `hm [Nseq, HV, K, V+K]` FP32，左 `[0,V)` 是 `h`（K×V）、右 `[V,V+K)` 是 `m`（K×K）；
* 三个舍入点：`h` 进 Cube 前降 BF16、`v_new` 进 Cube 前降 BF16、`m` 链 FP32（S4 用 FP32 原生 MMAD）。
