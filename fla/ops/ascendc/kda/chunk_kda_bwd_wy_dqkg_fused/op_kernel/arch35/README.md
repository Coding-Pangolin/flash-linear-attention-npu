# arch35 (Ascend950 / A5)

Dual-path kernel (same pattern as `chunk_bwd_dv_local`):

| Path | Gate | Sources |
|------|------|---------|
| 910B | `__CCE_AICORE__ != 310` | parent `*_common/_cube/_vector.h` |
| Ascend950 | `__CCE_AICORE__ == 310` | this directory |

- **common / cube**: thin includes of parent (`CATLASS_ARCH=3510` / `Ascend950`).
- **vector**: currently **thin-includes parent** (classic AscendC). A MicroAPI regbase
  body was tried (`chunk_kda_bwd_wy_dqkg_fused_regbase.h` kept for next land) but
  board runs hit **AICore 507015**; do not re-enable until masked MicroAPI loads
  are validated on 950 hardware.
- Host tiling shared. CMake adds `Ascend950PR_9599` when `ASCEND_COMPUTE_UNIT=ascend950`.

Build: `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd_wy_dqkg_fused`.
Board steps: [`../../ASCEND950_TEST_GUIDE.md`](../../ASCEND950_TEST_GUIDE.md).
