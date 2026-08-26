#!/bin/bash
set -e
cd "$(dirname "$0")"
bash gen.sh npu_custom.yaml
python3 setup.py bdist_wheel
pip3 install ./dist/fla_npu-1.0.0-*.whl --force-reinstall --no-deps

# The fla_npu runtime loads libcust_opapi.so only from the OPP tree embedded in
# the installed package (fla_npu/opp/vendors/fla_npu_transformer). The standalone
# wheel built here ships only the OPP skeleton, so importing fla_npu fails with
# FileNotFoundError once the external-vendor runtime fallback was removed (PR #322).
# Overlay the compiled custom OPP from the just-built fla-npu-*.run package into
# the installed package before any consumer imports fla_npu. Unlike main, the
# v26.6.0 run installer has no --install wheel-merge, so we install the OPP
# directly into the package-local opp/ tree.
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
opp_root="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/fla_npu/opp"
"$run_pkg" --quiet --install-path="$opp_root"
