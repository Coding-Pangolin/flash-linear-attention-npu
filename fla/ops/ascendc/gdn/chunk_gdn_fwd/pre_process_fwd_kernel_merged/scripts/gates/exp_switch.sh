#!/bin/bash
# 实验快路径（PPFM_OPT_ITERATION_PLAN.md §0.5）：
#   只编单算子 OPP 包（build.sh --pkg，~40 s，比整次 wheel 快 ~4 min），
#   把它装进「已安装包内」的 opp_vendors/<TAG>，再把 vendor 目录软链指过去。
#
# ⚠ 三条硬约束（实测，别照初版写）：
#   1. 实验目录最终解析路径必须落在 <site-packages>/fla_npu 之内，否则 import fla_npu 直接报错；
#   2. ln 不能覆盖真实目录 ⇒ 首轮必须先 rm -rf（切换目标已是软链时 ln -sfnT 才幂等）；
#   3. 切换后必须重启 Python 进程。
#
# 用法：  TAG=R17_xxx bash exp_switch.sh <soc> [repo_root]
# 还原：  TAG= 时把 vendor 软链指回包内原生目录（首次切换前会自动备份为 *.orig）
set -u
TAG=${TAG:?TAG=<本轮标识> 必填}
SOC=${1:-ascend950}
REPO=${2:-$(cd "$(dirname "$0")/../../../../../../.." && pwd)}
OP=pre_process_fwd_kernel_merged

cd "$REPO" || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PKG=$(python3 -c 'import fla_npu,os;print(os.path.dirname(fla_npu.__file__))')
V="$PKG/opp/vendors/fla_npu_transformer"
EXP="$PKG/opp_vendors/$TAG"
RUNBASE=/root/ppfm_exp/runs

echo "[exp] repo=$REPO pkg=$PKG tag=$TAG soc=$SOC"
echo "[exp] commit=$(git rev-parse --short HEAD)  $(git log --oneline -1 | head -c 80)"

# 1) 备份原生 vendor 目录（只做一次）
if [ ! -e "$V.orig" ] && [ ! -L "$V" ]; then
  cp -a "$V" "$V.orig" && echo "[exp] 备份原生 vendor -> $V.orig"
fi

# 2) 编单算子 OPP 包
rm -rf build_out
bash build.sh --pkg --soc="$SOC" --vendor_name=fla_npu --ops="$OP" > /tmp/exp_build_${TAG}.log 2>&1
rc=$?
echo "[exp] build.sh rc=$rc (log /tmp/exp_build_${TAG}.log)"
[ $rc -eq 0 ] || { tail -n 20 /tmp/exp_build_${TAG}.log; exit 1; }

mkdir -p "$RUNBASE"
RUN=$(ls -t build_out/*.run | head -n 1)
cp -f "$RUN" "$RUNBASE/${TAG}.run"
echo "[exp] run sha256=$(sha256sum "$RUNBASE/${TAG}.run" | cut -d' ' -f1)"

# 3) 解包出独立基线（用于 §5.5 核验，不依赖安装树）
rm -rf "$RUNBASE/${TAG}.extract"
"$RUNBASE/${TAG}.run" --extract="$RUNBASE/${TAG}.extract" --noexec > /dev/null 2>&1
echo "[exp] extract kernel .o md5:"
find "$RUNBASE/${TAG}.extract" -name '*.o' | head -n 3 | xargs -r md5sum

# 4) 装到包内实验目录并切软链
rm -rf "$EXP"
"$RUNBASE/${TAG}.run" --quiet --install-path="$EXP"
mkdir -p "$EXP/vendors"
echo "load_priority=fla_npu_transformer" > "$EXP/vendors/config.ini"
rm -rf "$V"
ln -sT "$EXP/vendors/fla_npu_transformer" "$V"
ls -ld "$V"
echo "[exp] 切换完成（记得重启 Python 进程；还原用 TAG= 的反向软链或 $V.orig）"
