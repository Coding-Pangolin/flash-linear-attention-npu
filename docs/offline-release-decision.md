# 离线发布方案决策对比：编译期三包源码 vs 运行时 Python 依赖

> 状态：待决策（explore 分支 `20260818_214900_offline-bundle-explore`）
> 日期：2026-08-20

## 1. 背景

`flash-linear-attention-npu` 的发行物是**预编译 wheel**：内嵌算子 OPP（即装即用），通过 ctypes 直调，不绑定特定 `torch_npu` 版本。

"离线"诉求目前分为**两个互相独立的层面**：

| 层面                                       | 依赖内容                                                               | 何时需要                                        | 是否已实现            |
| ------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------- | --------------------- |
| **A. 编译期 C/C++ 三包源码**         | abseil / protobuf / opbase / catlass / eigen / json / gtest / makeself | 开发者从源码`build.sh` / `pip wheel` 编译时 | ✅ 已实现（本次 B2）  |
| **B. 运行时 Python 依赖（PyPI 包）** | numpy / scipy / sympy / torch / torch_npu 等                           | 客户`pip install`/运行算子时                  | ❌ 未实现，本文档评估 |

**关键前提**：普通客户 `pip install` 预编译 wheel 后是**即装即用、不再编译**的，所以层面 A（编译用三包）对普通客户**无感**；只有需要自己编译的开发者才涉及。而层面 B 是所有使用场景都需要的。

---

## 2. 方案 A：离线编译期 C/C++ 三包源码

### 做法

把编译需要的 C/C++ 三包（**最小源码子集**，见下）打进 wheel 的 `fla_npu/offline/third_party/`。需要编译时用脚本提取到源码 `third_party/`（CANN_3RD_LIB_PATH），**编译全程不联网**。opbase/catlass 已实现"tar 源码（无 .git）"离线化。

**最小子集策略**（`scripts/tools/prepare_offline_bundle.py`）：
- header-only 库只带 include 头：json（`include/`）、eigen（`Eigen/`）、catlass（`include/`）
- abseil / protobuf 走 `ExternalProject`，只带 `pkg/*.tar.gz` **归档**（不再预置其完整源码目录），离线编译时由 CMake 解压 + patch + 编译
- makeself 带自身文件；opbase 带编译所需源码（剔除 `docs/` 等）

### 实测体积（本机 A2 + CANN 9.1.0 实测）

| 项                            | 瘦身前（完整目录） | 瘦身后（最小子集） |
| ----------------------------- | ------------------ | ------------------ |
| offline 三包解压体积          | ~88.8 MB           | **~30.7 MB**   |
| offline 三包压缩进 wheel 后   | ~21 MB             | **~12.3 MB**   |
| **打完三包后 wheel 文件大小** | ~40.3 MB           | **~26.0 MB**   |

瘦身后 `offline/third_party` 顶层构成（解压）：eigen ~11.4 MB、`pkg/` 归档 ~8.0 MB、opbase ~7.6 MB、catlass ~2.8 MB、json ~0.9 MB、makeself ~0.1 MB。

### 优点

- **wheel 增量小**（压缩后 ~12.3 MB），对多数"即装即用"客户影响可接受
- 彻底解决**编译现场网络依赖**——这是之前现场编译最痛苦的点
- 瘦身后仍通过**全链路离线编译**验证（断网 + `pip wheel --no-build-isolation`：abseil/protobuf 从归档解压编译，json/eigen/catlass/makeself 本地命中，产出新 wheel）
- 不改变运行时行为，不影响"与 torch_npu 版本解耦"的既定设计

### 缺点 / 局限

- **没有解决**普通客户的运行时依赖：客户仍需在线装 numpy 等 Python 包 + 自装 CANN/torch/torch_npu
- 只造福**需要编译**的开发者；对"纯使用"客户是纯占体积（压缩后多 ~12.3MB）
- 三包版本需随 wheel 固定（bundled 版本与源码匹配），上游版本变化需重新出包

### 明确不做的事

- 不内嵌 torch/torch_npu（见方案 B 分析）
- 不上 PyPI（C/C++ 源码包无法在 PyPI 合法分发）

---

## 3. 方案 B：包含运行时 Python 依赖（PyPI 包）离线

### 做法

把运行时 Python 依赖（numpy/scipy 等，甚至 torch/torch_npu）也随发行物提供，让客户**完全不联网**也能装。

### 实测体积（本机环境实测量）

| 依赖                                                                       | site-packages 体积 |
| -------------------------------------------------------------------------- | ------------------ |
| numpy                                                                      | 26.0 MB            |
| scipy                                                                      | 26.7 MB            |
| sympy                                                                      | 0.3 MB             |
| PyYAML                                                                     | 3.1 MB             |
| 其余小件（decorator/attrs/protobuf/psutil/expecttest/packaging/ml_dtypes） | ~0.5 MB            |
| **纯 Python 依赖小计**                                               | **~57 MB**   |
| torch                                                                      | 435 MB             |
| torch_npu                                                                  | 135 MB             |
| **若含 torch 系总计**                                                | **~630 MB**  |

### 子方案对比

| 子方案                      | 做法                                                                                       | 体积影响                      | 可行性                      |
| --------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------- | --------------------------- |
| B1：仅纯 Python 依赖离线    | `pip download -r requirements.txt` 到离线目录，`--no-index --find-links` 装            | +~57 MB（wheel 内或配套目录） | ✅ 可行，体积可控           |
| B2：连 torch/torch_npu 离线 | 把 torch(435M)/torch_npu(135M) 也打包                                                      | **+~630 MB，GB 级**     | ⚠️ 体积爆炸、版本矩阵爆炸 |
| B3：pip 离线源准备脚本      | 有网时一次`pip download --dest` 全部依赖到目录，客户 `--no-index` 离线装（不打进 whl） | 0（whl 不变）                 | ✅ 推荐                     |

### 优点

- 普通客户也能离线（至少纯 Python 依赖可离线）
- `pip download` 是标准 pip 机制，实现简单

### 缺点 / 重要的现实约束

1. **torch / torch_npu 是 GB 级且强版本绑定**
   - 我们刻意**不 pin torch_npu 版本**（前面确认"与版本不依赖"）；一旦做全离线，**必须固定 torch_npu/torch 版本**，反而破坏版本灵活性
   - 每种 `Python 版本 × torch 版本 × torch_npu 版本` 都需要一份离线包 → **组合爆炸**，不可持续
2. **Python 包不应内嵌第三方 Python 库**（违反打包惯例，转发许可/体积/冲突风险）
3. **wheel 膨胀**：若把 torch 系打进 wheel，从 62MB → **~700MB**，且仍需 CANN 自装（CANN 才是真正的系统级 .run）
4. **CANN 本身离线**：CANN toolkit 是系统级 `.run`（官方已支持离线安装），无法也不必打进 Python wheel

### 结论

- **纯 Python 依赖（numpy/scipy 等）离线**：可行、体积可控，建议通过 **B3（pip download 离线源脚本）** 提供，而不是硬塞进 wheel
- **torch / torch_npu 全离线**：因体积（GB 级）+ 版本矩阵爆炸 + 破坏版本灵活性，**不建议**打进 fla_npu wheel

---

## 4. 综合对比

| 维度                       | A：编译三包源码离线 | B1：纯 Python 依赖离线    | B2：连 torch/torch_npu 离线 |
| -------------------------- | ------------------- | ------------------------- | --------------------------- |
| 服务对象                   | 需编译的开发者      | 所有使用客户              | 所有使用客户                |
| wheel 体积增量             | ~12.3 MB           | ~57 MB（或 0 用 B3）      | ~630 MB（GB 级）            |
| 是否破坏"不 pin torch_npu" | 否                  | 否                        | **是（必须 pin）**    |
| 版本矩阵负担               | 低（三包固定即可）  | 中（每 python 一份）      | **极高（爆炸）**      |
| 编译现场离网               | ✅ 彻底解决         | 无关                      | 无关                        |
| 运行离网                   | ❌                  | ✅（纯 Python）           | ✅（含 torch）              |
| 与 CANN 离网交互           | 无冲突              | 仍须 CANN 自装            | 仍须 CANN 自装              |
| 实现复杂度                 | 已完成              | 简单（pip download 脚本） | 复杂且笨重                  |
| 合规/惯例                  | 三包为单独源码，OK  | OK                        | ⚠️ 违反惯例               |

---

## 5. 建议

**采用「A + B1/B3」组合**，覆盖两场景且不发散：

1. **A（编译三包离线）**：已完成。让"需编译的开发者"彻底离线编译。← 本次 B2 已交付
2. **B3（纯 Python 依赖离线源）**：提供一个 `scripts/tools/prepare_python_offline.py`（有网时 `pip download -r requirements.txt` 到目录）供客户 `--no-index --find-links` 离线装；**不打进 wheel**，保持 wheel 轻量。
3. **torch / torch_npu**：**不建议打进 fla_npu wheel**。它们的离线由客户基于固定的 torch_npu 适配组合自行用 `pip download` 准备，或依赖 CANN 官方离线安装流程。若要强离线，应作为**独立的超大离线源**（按版本矩阵单独发布），而非塞进 fla_npu wheel。

> 若业务上**必须**让所有客户（含 torch 系）真离线，应走 CANN 官方 .run（含 torch/torch_npu 的整机离线部署包），而不是把 GB 级依赖塞进 Python wheel。
