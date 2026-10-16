import io,os,shutil
old = "    if gk is not None and head_key != head_value:\n        raise ValueError(\"gk (KDA/DPLR) requires HK == HV (k must be the pre-gated kg).\")\n"
new = "    # KDA 的 gk 是按 value head 给的 [B,T,HV,K]（竞品 chunk_kda 文档：GVA 时 gate 形如 [B,T,HV,K]），\n    # k 仍按 HK 头、由 hv // (HK -> HV 的分组) 映射：HK < HV（GVA）合法，不再要求 HK == HV。\n"
files = ["/usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu/ops/ascendc/_aclnn_ctypes.py",
         "/workspace/bartonfang/flash-linear-attention-npu/torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py"]
for f in files:
    s = open(f, encoding="utf-8").read()
    if old in s:
        shutil.copyfile(f, f + ".bak_gva")
        open(f, "w", encoding="utf-8").write(s.replace(old, new))
        print("patched", f)
    else:
        print("pattern not found in", f)
