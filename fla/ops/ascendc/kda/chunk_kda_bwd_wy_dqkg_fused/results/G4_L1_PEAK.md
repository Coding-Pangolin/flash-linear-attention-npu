# G4 — L1 peak（WY A resident v2）

> Shape model：BT=64，bf16，`validRows=64`

| Region | Bytes | 说明 |
|--------|------:|------|
| Working L1A+L1B (Stage3 64×64) | 16 KB | masked + A（或 interim）|
| **A resident bank** | 8 KB | `L1_A_RESIDENT_OFF = 256 KB` |
| Stage1 worst (do×h etc.) | ≪ 256 KB | 不与 resident 并存于同 head 尾段 |
| L1 total | 512 KB | Ascend910B |

约束：Stage2 结束时 A 在 L1[0]；拷入 resident 后 Stage3 gemm1 可用低端 L1；gemm2 `skipLoadA` 读 resident。  
同窗 **per-head** `Stage2→Wait Mask→Stage3`，避免 head1 Stage2 冲掉 head0 的 A。
