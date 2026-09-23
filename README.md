# flash-linear-attention-npu

## 🔥Latest News

- [2026/09] torch_npu 解耦；新增算子：KDA 正反向（`recurrent_kda` / `chunk_kda_fwd`）、GDN 大融合（fused forward / backward finalize）。
- [2026/06] 发布 v26.6.0 预编译 wheel，覆盖 A2 / A3 / A5 目标，可在 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 下载。
- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

本仓不自动安装 `torch`、`torch_npu`、`torchnpugen`、`triton-ascend`，这些包必须与 CANN 与 Python 版本匹配，需要使用者按环境自行安装；版本不匹配时，构建或运行会报错。依赖匹配关系与检查方式见下文 Step 1 / Step 2。

## ⚡️快速上手

### Step 0. 确认硬件与目标芯片

在开始前，先确认机器上可用的 NPU 类型：

```sh
npu-smi info
```

确认机器类型后，按目标芯片选择后续构建参数（`--soc` / `FLA_NPU_SOC`）：

| 产品 | `--soc` / `FLA_NPU_SOC` |
| ---- | --------------------------- |
| A2   | `ascend910b`              |
| A3   | `ascend910_93`            |
| A5   | `ascend950`               |

### Step 1. 部署 CANN 开发环境

安装 toolkit 与对应机型 ops 两个包（A2/A3：CANN ≥ 8.5.2；A5：CANN ≥ 9.0.0），下载页：[CANN 社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann)

- `Ascend-cann-toolkit_<version>_linux-<arch>.run`
- `Ascend-cann-<chip>-ops_<version>_linux-<arch>.run`

### Step 2. 编译并安装 wheel

#### 2.1 环境检查

以下命令在已激活的 Python 环境（conda/venv）的仓库根目录执行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 每次进入新 shell / Docker / venv 都要重新执行

# 本仓不自动安装 torch / torch_npu / triton-ascend，需按 CANN 与 Python 版本自行安装

python -m pip install -r requirements.txt
python scripts/check_npu_env.py            # 无 NPU 的纯构建环境可加 --build-only
```

#### 2.2 编译

```sh
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py            # A2；A3→ascend910_93，A5→ascend950

# 可选：只构建指定算子；其余环境变量见开发者指南
FLA_NPU_OPS=chunk_fwd_o,chunk_bwd_dv_local FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

`FLA_NPU_SOC` 同时决定产物档位与发行名（A2→`a2`、A3→`a3`、A5→`a5`）。`dist/` 下可能同时存在
不同版本、不同档位的 wheel，安装时必须传入本轮构建输出的准确文件名，不要用通配符。其余与
构建、发布相关的环境变量见[开发者指南](docs/开发者指南.md) 场景 1 / 场景 6。

表外取值（例如拼错的芯片名）会让构建直接失败，不会产出一个名字对不上的包。需要给一次性产物
打标记时用 `FLA_NPU_WHEEL_BUILD_TAG`，它会把标签写进文件名；发布门禁会拒绝带 build tag 的产物。

#### 2.3 安装

```sh
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<wheel文件名>.whl
```

> 自编 wheel 带本地版本（如 `26.9.1+26.9.1.dev0a1b2c3`），排在已装的同档位正式包之上，上面的
> 命令仍带 `--force-reinstall`，重复构建同一版本时不会被 pip 当作"已是最新版本"跳过。

需要单独编译一个或多个算子 run 包的开发者场景见[开发者指南](docs/开发者指南.md) 场景 1。

#### 2.4 【可选】直接安装已发布的 wheel

官方 wheel 按产品档位发布到 PyPI，按机器芯片选一个包名安装即可（CANN 与 `torch` /
`torch_npu` / `triton-ascend` 仍要先按 Step 1 / Step 2 准备好；wheel 内嵌预编译 OPP 与离线
编译 bundle，但**不打包**这些运行时依赖）：

| 芯片 | 档位 | 安装 | 平台标签 |
| --- | --- | --- | --- |
| 910B（A2，`ascend910b`） | a2 | `python -m pip install flash-linear-attention-npu-a2` | `manylinux_2_34_aarch64` |
| A3（`ascend910_93`） | a3 | `python -m pip install flash-linear-attention-npu-a3` | `manylinux_2_34_aarch64` |
| 950（A5，`ascend950`） | a5 | `python -m pip install flash-linear-attention-npu-a5` | `manylinux_2_34_x86_64` |

档位写在包名里、架构写在 wheel 标签里：pip 只会看到与本机架构匹配的文件，架构不符的轮子在
`pip install` 阶段就被拒绝。wheel 内嵌的 OPP host 库与 Stable-ABI 薄层是按构建机架构编译的，
因此每组（档位 × 架构）都要有对应架构的构建机，上表是本轮已具备的三组；其余组合（x86_64 上
的 A2、aarch64 上的 A5）会在相应 runner 注册后随版本补发，包名不变。

本项目不做运行期芯片识别，同一架构换档位要按芯片换包名，装错档位会在调用算子时报错；各档位
是独立的 PyPI 项目，互不覆盖，可并排装在不同环境里。

**本地自编产物与 PyPI 包同名同标签**：910B 上执行
`FLA_NPU_SOC=ascend910b python scripts/build_wheel.py` 得到的就是
`flash_linear_attention_npu_a2-<版本>-py3-none-manylinux_2_34_aarch64.whl`，与
`pip install flash-linear-attention-npu-a2` 装到的是同一个发行名、同一个平台标签，两者互为
升级路径，不会在一个环境里留下两份互不知晓的 `fla_npu/`。版本号是两者唯一的差别：

| 出包类型 | 版本号 | 例（A2，分支 `v26.9.1`） |
| --- | --- | --- |
| 正式发布 | `<__version__>` | `26.9.1` |
| 每日 / 日常构建 | `<__version__>+<分支>_dev<commit7>` | `26.9.1+26.9.1.dev0a1b2c3` |

每日构建的本地版本取构建分支：`main` 上是 `26.7.0.dev0+main.dev0a1b2c3`，发布线 `v26.9.1` 上是
`26.9.1+26.9.1.dev0a1b2c3`（PEP 440 把下划线规范化为点号，`pip` 也按规范化后的形式比较）。
它既让每日包与正式包不会混为一谈，又排在**同版本正式包之上**，因此 `pip install` 一份每日
wheel 能覆盖已装的正式包。正式出包时用 `FLA_NPU_DISABLE_LOCAL_VERSION=TRUE` 关掉这个后缀
（发布工作流已带该开关），细节见[开发者指南](docs/开发者指南.md) 场景 6.1。

一份产物既不依赖 CPython ABI 也不依赖 libtorch C++ ABI，所以 `Requires-Python` 只有下限
`>=3.9`，不必按 Python / torch 小版本各发一份。wheel 声明 `torch>=2.7.1` /
`torch_npu>=2.7.1`：低于该版本时薄层无法加载，`import fla_npu` 会告警并自动回退 ctypes
实现（结果正确，只损失 host 侧加速），**不会中断导入**；离线或受控环境可用 `--no-deps`
安装，避免 pip 按 PyPI 上的 torch_npu 版本触发升级。

前置依赖下限（低于下限仍可 import，只给 RuntimeWarning；能否正常运行以实际环境为准）：

| 项 | 最低版本 | 说明 |
| --- | --- | --- |
| CANN（a2 / a3 档位） | 8.5.2 | |
| CANN（a5 档位） | 9.0.0 | 950 的 CANN 基线更高 |
| `torch` / `torch_npu` | 2.7.1 | torch_npu 从昇腾社区发布安装，PyPI 上的版本通常不可用 |
| `triton-ascend` | 3.2.0；CANN 9.x 需 ≥ 3.2.1 | 需与 CANN 版本匹配 |
| `glibc` | 2.34 | wheel 标签即 `manylinux_2_34_<arch>`，等于构建镜像（Ubuntu 22.04）的实测水位 |
| `libstdc++` | GLIBCXX 3.4.29 | 即 Ubuntu 22.04+ / GCC 11+；老系统上的加载失败见[离线编译与使用指南](docs/离线编译与使用指南.md) 第 7 节 |

运行期后端开关（默认已是 Stable-ABI 薄层，取值不识别时按默认处理）：

| 环境变量 | 取值 | 作用 |
| --- | --- | --- |
| `FLA_NPU_STABLE_ABI` | `ctypes` | 强制使用 ctypes 参考实现（默认优先薄层，薄层不可用时自动回退并告警一次） |
| `FLA_NPU_STABLE_VALIDATE` | `1` | 用 ctypes 参考实现做完整入参校验，结果与默认通路逐位一致 |
| `FLA_NPU_STABLE_TRACE` | `1` | 在 stderr 打印每个算子实际由哪个后端服务 |

卸载按发行名（档位）执行：

```sh
python -m pip uninstall -y flash-linear-attention-npu-a2   # 910B
python -m pip uninstall -y flash-linear-attention-npu-a3   # A3
python -m pip uninstall -y flash-linear-attention-npu-a5   # 950
```

> 从旧命名（不带档位的 `flash-linear-attention-npu`）升级过来的环境，先执行一次
> `python -m pip uninstall -y flash-linear-attention-npu`，否则新旧两个发行名会同时拥有
> `fla_npu/`，卸载其中一个会留下另一个的文件。

### Step 3. 验证与测试

```sh
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; print(hasattr(ascendc, 'chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

单算子测试（ATK 安装见 [Ascend/ATK](https://gitcode.com/Ascend/ATK)，用法见 [ATK 说明](tests/atk/README.md)）：

```sh
bash tests/atk/run_test_cpu.sh -op=<算子名> -npu_device_id=0
```

`-op` 可选值（即 `tests/atk` 下的算子目录名）：

- `causal_conv1d`
- `causal_conv1d_bwd`
- `chunk_bwd_dqkwg`
- `chunk_bwd_dv_local`
- `chunk_fwd_h`
- `chunk_fwd_o`
- `chunk_gated_delta_rule_bwd`
- `chunk_gated_delta_rule_bwd_dhu`
- `chunk_gated_delta_rule_bwd_finalize`
- `chunk_gated_delta_rule_fwd`
- `chunk_gated_delta_rule_fwd_h`
- `chunk_gated_delta_rule_fwd_prepare`
- `chunk_gdn_bwd_intra`
- `chunk_kda_bwd_recompute`
- `chunk_kda_fwd`
- `chunk_kda_fwd_finalize`
- `chunk_kda_fwd_prepare`
- `chunk_local_cumsum`
- `chunk_scaled_dot_kkt`
- `prepare_wy_repr_bwd`
- `prepare_wy_repr_bwd_da`
- `prepare_wy_repr_bwd_full`
- `recompute_w_u_fwd`
- `recurrent_gated_delta_rule`
- `recurrent_kda`
- `solve_tri`

## 开发者指引

开发者相关操作（单独编译单算子、一键编包、增加新算子、确认 wheel 来自最新源码）按场景拆分为独立文档；测试单算子和端到端验证见上文 Step 3：

- [开发者指南](docs/开发者指南.md)
- [在线 / 离线使用与编译指南](docs/离线编译与使用指南.md)（直接使用 wheel、在线编译后离线二次编译、全离线编译）

旧版本（v26.6.0 及更早）用户升级与兼容迁移见[兼容与迁移指南](docs/兼容与迁移指南.md)。

## 维护文档

- NPU CI 维护说明见 [`docs/Fla-npu仓CI部署教程.md`](docs/Fla-npu仓CI部署教程.md)。
- 旧版本用户升级与兼容迁移见 [`docs/兼容与迁移指南.md`](docs/兼容与迁移指南.md)。
- 开发者分场景指南见 [`docs/开发者指南.md`](docs/开发者指南.md)。

## 🔍目录结构

关键目录如下：

```
├── cmake                              # 项目工程编译目录
├── common                             # 项目公共头文件和公共源码
├── fla                                # 算子库核心包
│   └── ops
│       ├── ascendc                    # AscendC 算子实现
│       │   ├── common                 # 公共模块（GroupedMatMul 等）
│       │   └── gdn                    # GDN 算子
│       │       ├── chunk_gdn_fwd      # 前向传播算子
│       │       │   ├── chunk_fwd_h
│       │       │   ├── chunk_fwd_o
│       │       │   ├── chunk_gated_delta_rule_fwd_h
│       │       │   └── recompute_w_u_fwd
│       │       ├── chunk_gdn_bwd      # 反向传播算子
│       │       │   ├── chunk_bwd_dqkwg
│       │       │   ├── chunk_bwd_dv_local
│       │       │   ├── chunk_gated_delta_rule_bwd_dhu
│       │       │   ├── prepare_wy_repr_bwd_da
│       │       │   └── prepare_wy_repr_bwd_full
│       │       ├── gdn_preprocess     # 预处理算子
│       │       │   └── causal_conv1d
│       │       └── recurrent_gdn      # 推理算子
│       │           └── recurrent_gated_delta_rule
│       └── triton                     # Triton 算子实现
├── torch_custom                       # 自定义PyTorch算子适配
├── examples                           # 端到端算子开发和调用示例
│   └── flash_gated_delta_rule.py      # 完整GDN接入调用示例
├── scripts                            # 脚本目录，包含算子构建相关配置文件
├── docs                               # 文档目录（兼容迁移指南、开发者指南等）
├── tests                              # 测试工程目录
├── gdn-verify.sh                      # GDN 一键验证脚本
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
├── CONTRIBUTING.md                    # 贡献指南
├── SECURITY.md                        # 安全声明
├── LICENSE                            # 仓库级许可证说明
├── LICENSES                           # 许可证全文
├── NOTICE                             # 来源与再分发说明
└── requirements.txt                   # 本项目需要的第三方依赖包
```

## 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [NOTICE](NOTICE)

## ⚖️许可证说明

本仓库包含多种许可证文件：未在文件头或更具体说明中另行标识的原创代码使用 BSD 3-Clause License；从 CANN ops-transformer 改编的代码，以及文件头标识为 CANN Open Software License Agreement Version 2.0 的代码，使用 CANN Open Software License Agreement Version 2.0。该 CANN 许可证全文见 [LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt](LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt)，来源和再分发说明见 [NOTICE](NOTICE)。若文件级许可证说明与仓库级说明不一致，以文件级说明为准。

## 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
