# pre_process_fwd_kernel_merged 接入 Stable-ABI 适配层（交付说明）

> 适用：`fla_npu` 仓 `torch_custom/fla_npu/README.md` 要求的 **torch Stable ABI 适配层**
> （规范原文：`docs/architecture/适配层接入指南.md`）。
> ⚠ 本文档是按 **本地 9/23 快照**（`work\rbtry2691`）写的；仓里 HEAD 是 9/27，apply 前先跑
> §2 的门禁确认锚点/规范没变。

## 0. 这次要做什么（规范原文映射）

README §1.1：一次适配 = **新建 1 个算子文件 + 1 行 `#include` + 2 行注册 + 1 个 Python wrapper
+ 1 行 public 名**；并且 **不要求写 ctypes 适配**（ctypes 只是回退后端）。

| # | 位置 | 我们的内容 | 状态 |
|---|---|---|---|
| 1 | `csrc/src/stable_pre_process_fwd_kernel_merged.cpp` | `kSchema_pre_process_fwd_kernel_merged` + `run_npu_pre_process_fwd_kernel_merged` | ✅ 已写（本目录同名文件） |
| 2 | `csrc/src/stable_ops.cpp` | `#include` 一行 + `m.def` / `m.impl` 各一行 | ✅ 由 `apply_stable_abi.py` 自动插 |
| 3 | `fla_npu/ops/ascendc/_stable.py` | 真签名 wrapper（`_op(...)( ... , _current_stream_ptr())`） | ✅ 见 `_stable_wrapper_snippet.py` |
| 4 | `fla_npu/ops/ascendc/__init__.py` | `_ASCENDC_OPS` 加 `"npu_pre_process_fwd_kernel_merged"` | ✅ 由脚本插 |
| 5 | `MUTATED_ARGUMENTS` | **不需要**（hm 是新建输出，没有原地写参数） | — |

## 1. 关键契约（照抄规范，逐条对应我们的算子）

- **schema 形参顺序 == `run_` 形参顺序 == `FLA_STABLE_EXEC` 实参顺序 == aclnn 头文件顺序**，
  `stream` 固定在**最后**。我们的 aclnn 原型（`op_host/op_api/aclnn_pre_process_fwd_kernel_merged.h`）：
  `k, w, u, gOptional, gkOptional, bgOptional, vOptional, cuSeqlensOptional, chunkSize, hmOut`
- 取维度一律写 `SIZE_OF(meta, dim)`，**不要在 helper 里读维度**（否则报错里的张量名会变成 helper 的形参名）。
- host int 数组（我们的 `cu_seqlens`）：Python 侧 `_host_ints(...)`，C++ 侧 `int_values(...)` + `int_array(...)`。
- 输出：`allocate_sizes({Nseq, HV, K, V+K}, kFloat, k_meta)`，再 `out_tensor(meta_of(hm))`。
- stream：**只准** `_current_stream_ptr()` 现取，不许缓存（vLLM 多线程多 stream）；两条 accessor 的
  选择逻辑固定在 `_stable.py` 里，算子侧不感知。
- 非连续输入如实交给 `aclCreateTensor`，**不要**在适配层补 `.contiguous()`。

## 2. Apply（246 恢复后，约 1 分钟）

```bash
# 本目录 → 246（host）
scp -r work/impl/stable_abi admin123@192.168.13.246:/data/admin123/bartonfang/

# 容器内执行（repo 路径按实际）
docker exec admin123-gdn-test python3 /workspace/bartonfang/stable_abi/apply_stable_abi.py \
    --repo /workspace/bartonfang/flash-linear-attention-npu --dry-run
# 确认输出无误后去掉 --dry-run 再跑一次
```

脚本只做**文本插入**（幂等）：copy 算子文件、`stable_ops.cpp` 插 include + 2 行注册、
`_stable.py` 追加 wrapper、`__init__.py` 的 `_ASCENDC_OPS` 加一行。
若报“找不到锚点”，说明 9/27 的 `stable_ops.cpp` 与本地快照不同 —— 把 **`csrc/src/stable_ops.cpp`
和 `_stable.py` 头部 40 行**发我，我按现状改锚点。

## 3. 离线门禁（规范 §5；必须在编译前过）

```bash
cd /workspace/bartonfang/flash-linear-attention-npu
python torch_custom/fla_npu/tools/stable_coverage.py        # 覆盖 + 一算子一文件 + wrapper/schema/run_/注册 配对
python torch_custom/fla_npu/tools/op_abi_parity.py          # schema vs run_ vs ctypes 顺序
python torch_custom/fla_npu/tools/stable_ctypes_fallbacks.py # 适配层不许委派 ctypes（我们直接走 _op，没问题）
python -m unittest tests.test_stable_gates                   # 门禁自测（含 vendored header 校验）
```

`stable_coverage.py` 会按 **位置** 配对我们新加的 `m.def` / `m.impl` 两行（脚本已把两行都追加到各自
列表末尾），并检查 **wrapper 的位置参数个数 == schema 形参个数（10 个）且最后一个是 stream**。

## 4. 编译 + 设备回归

```bash
python torch_custom/fla_npu/csrc/build_stable.py --out /tmp/libfla_npu_stable.so --no-debug-probe
FLA_NPU_STABLE_LIB=/tmp/libfla_npu_stable.so PYTHONPATH=<env> \
    python tests/stable_abi/test_input_lifetime.py
```

**设备回归最低矩阵（规范 §6，按本算子形态取轴）**

| 轴 | 取值 |
|---|---|
| 序型 | `cu_seqlens=[0,T]` 单段 / 多段 `[0,88,188,256]` / **子区间** `[64,192]`（window 不是整根张量） |
| 可选参数 | `g` 单给 / `gk` 单给 / 都不给（应报错）/ 都给（应报错）/ `v=None` 与 `v=u` |
| GVA | `HK=2,HV=4`（`HK | HV`） |
| dtype | k/w/u/v `bf16`，`g`/`gk` `fp32`（bf16 gate 由 aclnn 层 Cast） |
| 边界 | `T=64`（单 chunk）/ 尾块不满 64（`cu=[0,88]`）/ 单 head `HV=1` / 空 `cu_seqlens`（应报错） |
| 错误路径 | `B != 1`、非递增 `cu`、`0 > cu[0]`、`cu[-1] > T`、`chunk_size != 64`、`bg != None` |

**关键：竞态类问题必须重复跑** —— 同一版本对 `cu=[0,256]` 连跑 ≥10 次，不允许出现任何一次 FAIL
（历史上这一条能一次 PASS、下一次 FAIL，单跑通过不等于修好）。

## 5. 模型 case（必跑，且要排在竞态收口之后）

```bash
# 输入：H20 导出的 case.pt（= benchmarks/cp/bench_pre_process_h20.py --case model-gk --precision ieee --save-io ./case_gk）
# 规模：T=11264 / HK=HV=32 / K=V=128 / BT=64 / varlen
# 对照：同一 case.pt 里 H20 的 default 输出（我们是 bf16 乘 + fp32 累加，结构上对应 default）
# 口径：atol=0.015  rtol=0.002  max_abs<=0.05，h / m 两半分别统计，要求 matched >= 0.999
```

顺序上必须：**先 6 形态 smoke + 敏感用例 ×10 收口竞态 → 再跑 model case**。
否则 model case 的失败里会同时混着“竞态”和“精度口径”两个原因，没法判读。

## 6. 已知风险 / 待确认

1. **仓库版本**：本适配按 9/23 快照写；仓 HEAD 是 9/27（`81a7346d`）。apply 前先跑 §3 门禁，
   若 `stable_coverage.py` 报锚点或规范不符，把现场文件发我改。
2. **`bg` / `chunk_size != 64`**：kernel 侧 TilingKey 3（DPLR）只注册未实现、`CV_BT=64` 写死。
   适配层已明确拒绝（`NotImplementedError` / `ValueError`），**建议同步给 ctypes wrapper 加同样守卫**，
   否则那条路会“静默算错”。
3. **K/V=256 分档**未实现（`K=V=128` 是当前唯一验收域）：schema 不限制，但调用会走不支持的 tiling —— 
   建议 wrapper 里先加 shape 断言，或明确记录为“未验收域”。
4. **ctypes 回退保留**：`_ASCENDC_OPS` 加了名字后，有 wrapper 就走适配层，没有才回落 ctypes；
   我们的 ctypes 实现在，属正常（`stable_ctypes_fallbacks.py` 只卡“适配层委派给 ctypes”，我们没委派）。
5. **性能验收**：规范要求 host 下发不劣化、不引入毫秒级开销。适配层是单次 launch（一次
   `FLA_STABLE_EXEC`），符合；但要实测一次 host 侧调用开销（vs ctypes 路径）留证。
