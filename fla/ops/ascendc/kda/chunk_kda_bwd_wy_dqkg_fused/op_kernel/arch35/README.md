# arch35 (Ascend950 / A5)

Dual-path kernel (same pattern as `chunk_bwd_dv_local`):

| Path | Gate | Sources |
|------|------|---------|
| 910B | `__CCE_AICORE__ != 310` | parent `*_common/_cube/_vector.h` |
| Ascend950 | `__CCE_AICORE__ == 310` | this directory |

- **common / cube**: thin includes of parent (WS/flag layout identical; `CATLASS_ARCH=3510` / `Ascend950` from shared common).
- **vector**: MicroAPI **regbase** rewrite (`chunk_kda_bwd_wy_dqkg_fused_regbase.h` + stage bodies). Schedule still mirrors DESIGN §5.1.
- Host tiling is shared (no `DAV_3510` A5 fork). CMake adds `Ascend950PR_9599` when `ASCEND_COMPUTE_UNIT=ascend950`.

Build: `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd_wy_dqkg_fused`.
