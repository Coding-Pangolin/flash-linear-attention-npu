#!/bin/bash
# 在 test 容器内运行：安装最新 wheel -> 精度冒烟 -> 计时
set -u
REPO=/workspace/bartonfang/flash-linear-attention-npu
WHL=$(ls -t "$REPO"/dist/*.whl | head -1)
echo "=== install: $(basename "$WHL") ($(date -r "$WHL" '+%F %T'))"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps "$WHL" 2>&1 | tail -2
echo "$WHL" > /tmp/ppfm_last_whl.txt
cd /tmp
echo "=== SMOKE ==="
python3 /tmp/npu_smoke_ppfm.py
echo "smoke_rc=$?"
echo "=== BENCH ==="
python3 /tmp/npu_bench_ppfm.py
echo "bench_rc=$?"
