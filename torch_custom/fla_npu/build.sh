#!/bin/bash
set -e
cd "$(dirname "$0")"
bash gen.sh npu_custom.yaml
rm -rf build dist flash_linear_attention_npu.egg-info fla_npu.egg-info
python3 setup.py bdist_wheel
shopt -s nullglob
wheels=(dist/flash_linear_attention_npu-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one flash-linear-attention-npu wheel, found ${#wheels[@]}." >&2
    exit 1
fi
python3 -m pip install "${wheels[0]}" --force-reinstall --no-deps --no-cache-dir

# The fla_npu runtime loads libcust_opapi.so from the OPP tree embedded in the
# installed package (fla_npu/opp/vendors/fla_npu_transformer). The standalone
# wheel only ships the OPP skeleton, so overlay the compiled custom OPP from the
# just-built run package into the installed package and refresh the wheel RECORD
# before any consumer imports fla_npu.
run_pkg=""
shopt -s nullglob
for cand in ../../build_out/fla-npu-*.run ../../build/fla-npu-*.run; do
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        run_pkg="$cand"
        break
    fi
done
shopt -u nullglob
if [ -z "$run_pkg" ]; then
    echo "[ERROR] No fla-npu-*.run package found to overlay the embedded OPP into the installed wheel." >&2
    exit 1
fi
chmod +x "$run_pkg"
"$run_pkg" --install --quiet
