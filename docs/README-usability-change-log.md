# README 易用性整改 · 修改点记录（21 点全量）

> 协作方式：每确认一个修改点 → 记入本表 → 确认下一个 → 最后一次性实施。
> 优先级说明：P0 = 事实性错误 / 会让用户操作失败；P1 = 核心需求缺失；P2 = 易用性增强。
> **状态总览（2026-08-07 14:30）**：21 点已全部实施并提交；PR #280 评审意见 10 条已全部修订并登记（见文末"## G. PR #280 评审意见修订"）。分支 `20260806_204500_docs-README-usability`（基于最新 main `ac46f1c3`，含 PR #274），已推送至 origin（Coding-Pangolin）。
> 口径：PR #274 已合入 main（`ac46f1c3`）。下文凡涉及"增量构建移除、`import fla_npu` 即加载、`scripts/check_install_workflows.py`、卸载说明、shell 环境钩子"等，均为 #274 在 main 上已有的内容；本次 PR 的增量是文档修正与新章节，与 #274 语义兼容（冲突合并时保留 #274 侧内容 + 叠加本次修正）。

---

## A. 根 README.md

### 修改点 #1（P0）：修正 `check_npu_env.py --build-only` 描述并补充完整预检方法（含 A1）

- **状态**：已实施
- **涉及**：`README.md` Step 2（原"如果依赖缺失，预检和一键编包都会在真正编译前失败，并列出缺失项……"）
- **依据（实测）**：`python scripts/check_npu_env.py --build-only` 在缺 torch 时也通过（跳过 torch/torch_npu/torchnpugen/triton 检查，`EXIT=0`），原文描述与行为不符；且原文档未告知如何判断 torch 系依赖是否完整、版本是否匹配。
- **实际改法**：明确 `--build-only` 只检查构建纯 Python wheel 所需环境（Python / bash / CANN），不检查 torch 系依赖；新增完整预检命令 `python scripts/check_npu_env.py`，说明其检查 `torch` / `torch_npu` / `triton-ascend` 是否可导入、版本下限与 NPU 可用性（`torch.npu.is_available()`），并说明无 NPU 卡的纯构建环境 `is_available()` 为 `False` 属预期、此时用 `--build-only` 即可；torch 系依赖缺失或版本不匹配时 `pip wheel` 会在构建/打包阶段报错，需按 CANN 与 Python 版本匹配的列表先行安装。

### 修改点 #2（P0）：移除已废弃的增量构建描述

- **状态**：已实施
- **涉及**：`README.md` Step 2 方式 A（原"真增量构建"段落 + 环境变量表 `FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_OPS` 两行）
- **依据**：PR #274 移除这两个开关，改为全量重建；只定位单算子用 `bash build.sh --ops=<op>`。
- **实际改法**：环境变量表删除 `FLA_NPU_INCREMENTAL_BUILD`、`FLA_NPU_OPS` 两行。PR #274 在 setup.py 中**一并移除了 `FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL`**（`tests/test_wheel_environment.py` 有守卫断言二者不得出现在 setup.py），因此这两行也随 #274 删除；最终环境变量表仅保留 `FLA_NPU_SOC` / `FLA_NPU_DISABLE_LOCAL_VERSION`（与 ac46 一致）。合并冲突时，"全量构建"段落采用 #274 在 main 上已有的版本（含"清理 `build/`/`build_out/`/`output/` 中间产物"及"dist 下可能多版本 wheel、需用准确文件名"的提示），不再自行改写。

### 修改点 #3（P0）：修正 Step 4 验证命令，区分新旧 wheel 行为（含 A2）

- **状态**：已实施
- **涉及**：`README.md` Step 4（原 `is_legacy_torch_ops_loaded()` + `hasattr(torch_npu.ops, 'chunk_fwd_o')` 验证）
- **依据（实测）**：`torch_npu.ops` 兼容入口行为随 wheel 版本而异——**当前仓库源码（PR #274 后）** 下 `torch_npu.ops` 属性不存在，旧命令抛 `AttributeError`；**fzy 安装的旧 wheel（26.7.0.dev0，2026-07-13 构建）** 导入 `fla_npu.ops.ascendc` 时自动调用 `install_torch_npu_ops_compat()`，`hasattr` 返回 `True`。
- **实际改法**：验证命令改为 `python -c "import fla_npu; print('ok')"` + `python scripts/check_packaged_wheel_api.py`；Step 4 新增两个 bullet 区分新旧 wheel 行为：新版默认不注册 `torch_npu.ops.*`，旧命令抛 `AttributeError` 属预期行为、不要用它验证新版，需要时先显式调用 `install_torch_npu_ops_compat()`；旧版（2026-07 之前的中间版本）导入即自动挂载、`hasattr` 返回 `True`，属旧版行为、不代表新 wheel。**实测补充（2026-08-07）**：`install_torch_npu_ops_compat()` 的调用必须先导入子模块——`import fla_npu` 后直接写 `fla_npu.ops.ascendc.install_torch_npu_ops_compat()` 会报 `AttributeError: module 'fla_npu' has no attribute 'ops'`（顶层 `__init__.py` 不自动导入 `ops`），必须 `from fla_npu.ops import ascendc`（或 `import fla_npu.ops.ascendc`）后再调用；已按此修正 Step 4 说明，给出可复制的 Python 片段。合并冲突时，Step 4 同时保留 #274 已加入的"卸载说明"（`pip uninstall flash-linear-attention-npu` 与 RECORD 无残留说明）及 `scripts/check_install_workflows.py` 看护脚本用法，与本次修正的验证命令共存。

### 修改点 #4（P1）：安装命令使用精确文件名并强制覆盖同版本号

- **状态**：已实施
- **涉及**：`README.md` Step 3 方式 A 与方式 B（原通配符 `dist/flash_linear_attention_npu-*.whl`）
- **依据**：通配符可能匹配多个 wheel 产物导致安装失败；重新构建 wheel 的版本号可能与已安装旧 wheel 相同（如本地 dev 版本 `26.7.0.dev0`），不带 `--force-reinstall` 的 `pip install` 会认为"已是最新版本"而跳过，导致实际仍是旧代码。
- **实际改法**：改为 `WHEEL_PATH="dist/<准确wheel文件名>.whl"` + `python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"`；方式 B 同理（`torch_custom/fla_npu/dist/<准确wheel文件名>.whl`）。新增引用块说明：版本号相同时需 `--force-reinstall` 强制覆盖，或先 `python -m pip uninstall -y flash-linear-attention-npu` 再装。

### 修改点 #5（P1）：`set_env.sh` 保留默认路径 + 补充自定义路径提示

- **状态**：已实施
- **涉及**：`README.md` Step 1（`export INSTALL_PATH` 注释 + `source $INSTALL_PATH/ascend-toolkit/set_env.sh` 后）
- **用户原话**：保留 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 作为默认，补充"若安装在自定义路径请 source 实际路径下对应的 set_env.sh"，但**文档中不出现具体自定义路径示例**。
- **实际改法**：Step 1 命令注释改为"设置需要安装的路径（请替换为实际安装路径）"，代码块后新增引用块：自定义路径时设置 `INSTALL_PATH` 为实际安装路径并 source 对应 `set_env.sh`；每次进入新 shell（Docker/Conda/venv）需重新 source。

### 修改点 #6（P1）：新增"从旧版本升级"章节，并补充新旧版本行为差异（含 B1/B2/B3）

- **状态**：已实施
- **涉及**：`README.md` 新增 `## 开发者指引 > ### 从旧版本升级（v26.6.0 及更早 → 最新）`（置于 Step 4 之后）
- **依据**：用户需求 #2/#3——`torch.ops.npu.*` 只支持到 v26.6.0，后续需 `fla_npu.ops.ascendc`；升级用户需要知道旧版与新版的构建、验证、兼容入口行为差异。
- **实际改法**：5 步——卸载旧包并清理残留（含 `custom_aclnn_extension_lib*.so` / 自定义 `libopapi.so`）→ 安装新版本 wheel → 迁移调用（旧→新对照表：`torch.ops.npu.npu_chunk_fwd_o` → `from fla_npu.ops.ascendc import chunk_fwd_o` 等）→ 验证（`check_packaged_wheel_api.py` + `test.sh --op gdn_fwd_o`）→ 迁移期临时兼容（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()`，注明 legacy 需 `FLA_NPU_BUILD_LEGACY_EXTENSION=1`，新代码勿用 legacy；**实测补充（2026-08-07）**：调用 `fla_npu.ops.ascendc.install_torch_npu_ops_compat()` 前必须先 `from fla_npu.ops import ascendc`，`import fla_npu` 后直接全限定名调用会报 `AttributeError`，已修正描述）。章节末尾新增"旧版本（≤ v26.6.0）与新版的主要行为差异"四项：
  - **B1 构建环境变量**：旧版支持 `FLA_NPU_INCREMENTAL_BUILD`（增量构建）、`FLA_NPU_OPS`（单算子 wheel）、`FLA_NPU_SKIP_RUN_BUILD` / `FLA_NPU_SKIP_RUN_INSTALL`（run 包控制）；新版（PR #274 起）全部移除，统一全量构建（自动清理 `build/`/`build_out/`/`output/` 中间产物），单算子定位改用 `bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=<op>` 构建 run 包，旧脚本中的 `FLA_NPU_INCREMENTAL_BUILD=1` / `FLA_NPU_OPS=...` 需要删除。
  - **B2 验证方式**：旧版 Step 4 的 `fla_npu.is_legacy_torch_ops_loaded()` 与 `hasattr(torch_npu.ops, ...)` 在新版不再适用；统一改用 `python -c "import fla_npu; print('ok')"` + `python scripts/check_packaged_wheel_api.py`。
  - **B3 `torch_npu.ops` 挂载行为**：旧版 wheel（2026-07 之前的中间版本）导入 `fla_npu.ops.ascendc` 即自动挂载 `torch_npu.ops.*`；新版（PR #274 后构建）默认不挂载，迁移期需显式调用 `install_torch_npu_ops_compat()`。
  - **`test.sh` 算子名**：`recompute_wu_fwd` 在新版统一为 `recompute_w_u_fwd`。

### 修改点 #7（P1）：新增"在 torch_custom 新增 Python 接口"指引

- **状态**：已实施
- **涉及**：`README.md` 新增 `### 在 torch_custom 新增 Python 接口`（位于升级章节后）
- **依据**：用户需求 #3——torch_custom 下怎么加新接口。
- **实际改法**：4 步核心链路——`_aclnn_ctypes.py` 新增 `npu_xxx(...)` wrapper → `__init__.py` 的 `_ASCENDC_OPS` 注册（自动导出 `npu_xxx` 与去前缀短名）→ 新增 `test_npu_<op>.py` 并接入 `test.sh` → 重新构建 wheel/run 包并验证；链接到 `torch_custom/fla_npu/README.md` 详情。

### 修改点 #8（P1）：新增"全新环境快速上手"前置步骤与冒烟命令

- **状态**：已实施
- **涉及**：`README.md` 新增 `Step 0. 确认硬件与目标芯片`（Step 1 之前）；Step 4 末尾补冒烟测试
- **依据**：用户需求 #1——全新环境按文档能否跑通。
- **实际改法**：Step 0 用 `npu-smi info` 确认机器类型 + A2/A3/A5 ↔ `--soc`/`FLA_NPU_SOC` 对照表；Step 4 末尾补真实算子冒烟 `cd torch_custom/fla_npu/test && bash test.sh --device 0 --op gdn_fwd_o`。

### 修改点 #9（P2）：补全 `test.sh --op` 可选值

- **状态**：已实施
- **涉及**：`README.md` "测试单算子"可选值列表
- **依据**：实测 `test.sh` 含 `chunk_local_cumsum`、`chunk_scaled_dot_kkt` 两个已接入任务，原列表缺失。
- **实际改法**：列表补上 `chunk_local_cumsum`、`chunk_scaled_dot_kkt`，共 11 个，与 `test.sh` 逐条核对一致。

### 修改点 #10（P2）：精简方式 B 的 legacy 编译命令

- **状态**：已实施
- **涉及**：`README.md` Step 2 方式 B（原含 `FLA_NPU_BUILD_LEGACY_EXTENSION=1 bash gen.sh npu_custom.yaml` 与 `setup.py bdist_wheel` 两条）
- **依据**：legacy extension 是迁移期可选能力，放在主路径会让使用者误以为必须编译。
- **实际改法**：从方式 B 删除这两条命令；legacy 说明保留在"从旧版本升级"章节的迁移期临时兼容条目中。

### 修改点 #11（P2）：修正 CANN 下载段文案

- **状态**：已实施
- **涉及**：`README.md` Step 1（原"推荐使用是社区版8.5.2，总共要下2个run包"）
- **依据**：原句语法错误；A2/A3/A5 需下载对应 ops 与 toolkit 包。
- **实际改法**：改为"推荐社区版 8.5.2，总共需要下载 2 个 run 包。这里以 A3 机器为例（即需要下载 A3-ops 与 toolkit），A2 / A5 机器请下载对应的 ops 与 toolkit 包。"

### 修改点 #12（P2）：概述补充依赖版本匹配说明

- **状态**：已实施
- **涉及**：`README.md` 概述段末尾
- **依据**：本仓不自动安装 torch 系依赖，需明确版本匹配要求。
- **实际改法**：新增一段——"本仓不自动安装 `torch`、`torch_npu`、`torchnpugen`、`triton-ascend`，这些包必须与 CANN 与 Python 版本匹配，需要使用者按环境自行安装；版本不匹配时，构建或运行会报错。"

---

## B. torch_custom/fla_npu/README.md

### 修改点 #13（P1）：新增"导入契约"章节

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` 开头（默认交付目标段后）
- **依据**：`import fla_npu` 即定位并加载 `libcust_opapi.so`，用户需要知道导入前提与常见失败原因。
- **实际改法**：新增章节说明——导入/构建前先 source CANN `set_env.sh`（默认路径 + 自定义路径提示）；以表格列出常见现象与处理：standalone wheel 缺 OPP（需先装 run 包或设 `FLA_NPU_OPP_PATH`）、未 source set_env.sh 导致 dlopen 报错、安装 run 包后需重启 Python 进程。

### 修改点 #14（P1）：修正"新算子如何接入默认 runtime"

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "新算子如何接入默认 runtime"步骤 6/7
- **依据**：原步骤缺少 mutation 契约与正反向绑定细节，`MUTATED_ARGUMENTS` / `BACKWARD_OPS` 实际存在于 `__init__.py`。
- **实际改法**：步骤 6 明确 `BACKWARD_OPS` 映射（例 `causal_conv1d` → `causal_conv1d_bwd`）与 autograd 自动绑定；新增步骤 7 说明就地修改输入 tensor 的算子需在 `MUTATED_ARGUMENTS` 登记参数名（ctypes 直写 storage 时 PyTorch 无法自动发现副作用），并指向 `test/test_ascendc_mutation_contract.py`；原步骤 7 顺延为 8。

### 修改点 #15（P2）：构建/安装命令对齐 PR274

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "构建和验证默认 runtime" 三处安装命令
- **依据**：与根 README 修改点 #4 一致；`scripts/check_install_workflows.py` 已随 PR #274 合入 main。
- **实际改法**：三处 wheel 安装均改为 `--force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"`（`WHEEL_PATH` 用准确文件名）；在"构建和验证默认 runtime"末尾补充 `scripts/check_install_workflows.py` 使用说明（安装流程看护，CI 自动运行，标注随 PR #274 引入）。

### 修改点 #16（P2）：legacy 章节补充迁移指引

- **状态**：已实施
- **涉及**：`torch_custom/fla_npu/README.md` "legacy torch_npu / torch.ops.npu 路径" 章节末尾
- **依据**：与根 README 升级章节呼应，明确 legacy 支持边界。
- **实际改法**：末尾新增——"`torch.ops.npu.*` / `torch_npu.ops.*` 只支持到 v26.6.0，从旧版本迁移到最新版本的完整步骤见根 README 的'从旧版本升级'章节；新代码请勿使用 legacy 路径。"

---

## C. AGENTS.md

### 修改点 #17（P2）：移除增量构建命令并修复多余代码围栏（含 A3）

- **状态**：已实施
- **涉及**：`AGENTS.md` "构建命令"（原 `FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_OPS` 两条命令）
- **依据**：与根 README 修改点 #2 一致（PR #274 已合入 main）；实测发现"单算子 run 包命令"代码块后多出一个 ``` 围栏，导致后续 markdown 渲染异常。
- **实际改法**：删除两条增量/单算子 wheel 构建命令，改为"源码或适配修改后仍执行完整 wheel 构建；构建流程会清理上一轮 `build/`、`build_out/`、`output/` 中间产物，不再支持增量构建。只定位单算子时，用 `bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=<op>` 构建单算子 run 包"（单算子产物不能替代完整 wheel 的全量重编），并给出示例命令；同时删除构建命令后多余的 ``` 代码围栏，修复 markdown 渲染。

### 修改点 #18（P2）：`set_env.sh` 硬编码补自定义路径提示

- **状态**：已实施
- **涉及**：`AGENTS.md` "构建命令"环境准备段
- **依据**：与根 README 修改点 #5 一致。
- **实际改法**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 上行加注释"CANN 安装于自定义路径时，请替换为实际路径下对应的 set_env.sh"。

---

## D. examples/README.md

### 修改点 #19（P2）：重写为实际样例

- **状态**：已实施
- **涉及**：`examples/README.md` 全文
- **依据**：原文件为 CANN ops-transformer 模板——引用不存在的 `mc2/`、失效链接 `../docs/zh/develop/aicore_develop_guide.md`，与本仓实际样例脱节。
- **实际改法**：重写为——简介（三个实际样例：`flash_gated_delta_rule.py`、`add_example/`、`fast_kernel_launch_example/`）→ 目录说明 → 快速运行（GDN 端到端、add_example、fast_kernel_launch_example）→ 新增示例要求（独立可运行、配单算子测试并接入 test.sh、优先 `fla_npu.ops.ascendc` / `.triton` 稳定入口、提供 CMakeLists.txt 参与统一编译）。

---

## E. CONTRIBUTING.md

### 修改点 #20（P2）：补充"本仓特有的贡献要求"

- **状态**：已实施
- **涉及**：`CONTRIBUTING.md` "贡献新算子"第 4 步之后
- **依据**：用户需求 #3/#4——新增算子必须提供稳定 Python 入口。
- **实际改法**：新增章节——新增算子必须提供 `fla_npu.ops.ascendc` 稳定 Python 入口，不得仅以 legacy `torch.ops.npu.*` / `torch_npu.ops.*` 交付；交付内容含 `_aclnn_ctypes.py` wrapper、`_ASCENDC_OPS` 注册（需要时同步 `BACKWARD_OPS` / `MUTATED_ARGUMENTS`）、`test_npu_<op>.py` 并接入 `test.sh`。

---

## F. .github/pull_request_template.md

### 修改点 #21（P2）：`set_env.sh` 硬编码补自定义路径提示

- **状态**：已实施
- **涉及**：`.github/pull_request_template.md` 验证方法 > 环境确认
- **依据**：与根 README 修改点 #5 一致。
- **实际改法**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 上行加注释"默认 CANN 安装路径；自定义安装路径时替换为实际路径下对应的 set_env.sh"。

---

## 实施后可实测项验证结果

| 验证项                                                                                    | 结果                                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python scripts/check_npu_env.py --build-only`（缺 torch）                              | 通过，`EXIT=0`，跳过 torch 系检查（印证 #1）                                                                                                            |
| 完整预检`python scripts/check_npu_env.py`（fzy 环境）                                   | 可检查 torch/torch_npu/triton-ascend 与`is_available()`（#1）                                                                                           |
| Step 4 新旧 wheel 行为（fzy 实测）                                                        | 旧 wheel（26.7.0.dev0）`hasattr` 返回 `True`；新版源码行为为抛 `AttributeError`（#3）                                                               |
| `bash test.sh --device 0 --op gdn_fwd_o --mode dry-run`                                 | 正常打印命令（印证#8/#9）                                                                                                                                 |
| `bash build.sh --help`                                                                  | `--pkg/--soc/--ops/--vendor_name` 参数存在                                                                                                              |
| `scripts/check_install_workflows.py`                                                    | 已随 PR#274 合入 main（修改点 #15 引用了其用法）                                                                                                          |
| `FLA_NPU_OPP_PATH` 环境变量                                                             | `fla_npu/__init__.py`、`install_opp.py` 中真实支持（修改点 #13）                                                                                      |
| `install_torch_npu_ops_compat()` / `load_legacy_torch_ops()`                          | `fla_npu/ops/ascendc/__init__.py`、`fla_npu/__init__.py` 中真实存在（#3/#6）                                                                          |
| `install_torch_npu_ops_compat()` 调用方式（fzy 实测）                                   | `from fla_npu.ops import ascendc` 后调用可挂载 `torch_npu.ops`；`import fla_npu` 后全限定名调用报 `AttributeError`，已按正确写法修正 Step 4（#3） |
| `_GET_WORKSPACE_ARGTYPES` / `BACKWARD_OPS` / `MUTATED_ARGUMENTS`                    | 代码中真实存在（修改点#14）                                                                                                                               |
| AGENTS.md 代码围栏修复                                                                    | `git diff` 确认删除多余 ```（#17）                                                                                                                      |
| 新增文档链接目标                                                                          | 全部有效                                                                                                                                                  |
| 残留校验（`FLA_NPU_INCREMENTAL_BUILD` / `FLA_NPU_OPS` / `mc2` / `../docs/zh` 等） | 本次改动文档中无残留                                                                                                                                      |
| `git diff --check`                                                                      | 通过                                                                                                                                                      |
| PR 冲突检查（compare API）                                                                | 分支基于`ac46f1c3`（main），无冲突                                                                                                                      |

**未实测**：需要真实 NPU + torch 环境才能执行的算子冒烟（`bash test.sh --device 0 --op gdn_fwd_o` 实际执行），本机缺 torch 环境，未执行。

---

## G. PR #280 评审意见修订（2026-08-07）

PR #280 上 reviewer 共提出 10 条行内意见，已逐条修订并登记如下。

### 意见 1/2（README Step 1）：CANN 版本推荐改最新稳定版 + 稳定链接

- **原状**：Step 1 写死"推荐社区版 8.5.2"，下载链接指向 8.5.2 具体版本页。
- **意见**：① 要求 8.5.2 以后版本，推荐使用最新社区稳定版本；② 更换成稳定版本链接。
- **实际改法**：文案改为"推荐使用最新的社区稳定版本（不低于 8.5.2，如需使用更新版本请参考 `check_npu_env.py` 支持的 CANN / torch_npu 版本组合）"；下载链接改为社区 CANN 下载总入口 `https://www.hiascend.com/developer/download/community`，并在其中选择最新的稳定版本。

### 意见 3（README Step 1）：修正 A3 写死的问题

- **原状**：安装命令写死 `./Ascend-cann-A3*run`，A2/A5 用户照抄会失败。
- **意见**：A3 写死的需要修正。
- **实际改法**：改为 `./Ascend-cann-<机器型号>*run`，注释说明"toolkit 与机型对应的 ops 包都必须安装，`<机器型号>` 请替换为实际机型对应的包前缀"，并给出 A3 → `Ascend-cann-A3*run`、A2 → `Ascend-cann-910b*run`、A5 → `Ascend-cann-950*run` 示例（已对照昇腾社区 CANN 官方 run 包命名核实）。

### 意见 4（README Step 2）：`--build-only` 环境不全补充 + 优先推荐完整预检

- **原状**：仅说明 `--build-only` 不检查 torch 系依赖。
- **意见**：① `--build-only` 检查的环境可能不全（CMake 等组件未检查），通过也可能不能正常编译，需补充支持的版本；② 优先建议用户用同时检测执行与编译的（不带 `--build-only`）。
- **实际改法**：改为"**建议优先使用完整预检（不带 `--build-only`）**"开头；补充说明 `--build-only` 不覆盖 `CMake`、编译器（`g++` / `bisheng`）、Python 头文件、`ninja` 等编译组件，检查通过不代表一定可以编译，缺失的编译组件会在 `pip wheel` 阶段才暴露。

### 意见 5（README Step 4）：去掉 PR 号，改描述 v26.6.0 后不维护兼容接口

- **原状**：Step 4 兼容入口说明写"新版 wheel（从当前仓库源码构建，PR #274 之后）"。
- **意见**：不应该显示写 PR 号，增加描述 v26.6.0 后不维护旧版本的兼容接口。
- **实际改法**：README 正文不再出现 PR 号；Step 4 改为"`torch.ops.npu.*` / `torch_npu.ops.*` 是旧版本（v26.6.0 及更早）的调用方式，**v26.6.0 之后不再维护旧版本兼容接口**，新代码请使用 `fla_npu.ops.ascendc` 下的稳定 Python 入口"，兼容细节（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()` 用法、`hasattr` 版本差异）移入新文档。

### 意见 6/10（README Step 4）：兼容性内容拆分到独立文档

- **原状**：Step 4 用大段篇幅说明 `torch_npu.ops.*` 兼容入口的新旧行为差异；方案 B / legacy 说明分散在主流程。
- **意见**：① README 主要写主流程，兼容性的篇幅太大，单起一个文档更合适；② 方案 B 是 legacy 路径，legacy 路径都单独放，引导用户用新方式。
- **实际改法**：新建 `docs/migration-guide.md`（兼容与迁移指南），收纳：调用方式演进对照表、从旧版本升级完整步骤、迁移期临时兼容（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()` 及 legacy 构建命令）、新旧版本行为差异、`hasattr(torch_npu.ops, ...)` 版本差异注意事项。README Step 4 与"开发者指引"只保留主流程 + 指向该文档的链接。

### 意见 7（README Step 4）：冒烟测试用算子名

- **原状**：冒烟测试 `bash test.sh --device 0 --op gdn_fwd_o` 未说明 `--op` 参数含义。
- **意见**：冒烟测试可以保留，但应该用算子名。
- **实际改法**：补充说明"`--op` 后跟算子名，`gdn_fwd_o` 为示例"。

### 意见 8（README 开发者指引）：开发者部分分场景拆分 + 单起 md

- **原状**："开发者指引"下按文档章节堆叠：从旧版本升级、在 torch_custom 新增接口、测试单算子、算子调用方式参考、端到端验证。
- **意见**：开发者下面分为多个场景区分——单独编译单算子（`bash build.sh` 方式、一键编包单算子）、增加一个算子的方式（目录结构、torch_custom），尽量对开发者透明，单起一个 md。
- **实际改法**：新建 `docs/developer-guide.md`（开发者指南），按场景拆分：场景 1 单独编译单算子（run 包）、场景 2 一键编包单算子、场景 3 增加一个新算子（目录结构 + torch_custom）、场景 4 测试单算子、场景 5 端到端 Example/ST 验证。README"开发者指引"精简为场景导航列表 + 链接。

### 意见 9（README 开发者指引）：如何确认新编版本来自最新源码

- **原状**：未说明如何确认新编的 wheel 确实由最新修改的源码编译。
- **意见**：怎么确定新编的版本是最新修改的源码编译的（比如开 debug 日志确定走的版本），需要找个方案。
- **实际改法**：在 `docs/developer-guide.md` 新增"如何确认新构建的 wheel 来自最新源码"章节，给出三种方式：方式 1 核对 wheel 文件名与版本号（构建日志文件名 → `pip show` / `importlib.metadata` 核对）；方式 2 确认实际加载的 OPP 路径（`import fla_npu` 后打印 `fla_npu.__file__` 与 `FLA_NPU_OP_API_LIB` / `ASCEND_CUSTOM_OPP_PATH` 环境变量，确认指向新 wheel 内嵌 OPP）；方式 3 修改后强制覆盖安装（`--force-reinstall`，或临时打印标记确认后移除）。
