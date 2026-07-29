# C1 baseline (I1+I2 on, pre Vec-bound knives)

Shape: B1 H=HV=32 T8192 K128 V128 BT64 bf16, state_v_first=false

Task Duration samples_us=[21476.148438]
med_ms=21.476 min=21.476 max=21.476
row0: cube=0.013640 aiv_scalar=NA aiv_mte2=NA wait2=9715.664062 wait4=8706.432617 wait6=920.503906
row1: cube=NA aiv_scalar=0.505723 aiv_mte2=0.356919 wait2=NA wait4=NA wait6=NA
row2: cube=NA aiv_scalar=0.136999 aiv_mte2=0.050299 wait2=NA wait4=NA wait6=NA
row3: cube=0.013640 aiv_scalar=NA aiv_mte2=NA wait2=9706.811523 wait4=8849.667969 wait6=934.930542
row4: cube=NA aiv_scalar=0.508837 aiv_mte2=0.360779 wait2=NA wait4=NA wait6=NA
row5: cube=NA aiv_scalar=0.140194 aiv_mte2=0.050416 wait2=NA wait4=NA wait6=NA

Verdict: Vec-bound. AIC waits V_GATE(id2)~9.7ms, V_MASK(id4)~8.7ms. Proceed V1-V3.
