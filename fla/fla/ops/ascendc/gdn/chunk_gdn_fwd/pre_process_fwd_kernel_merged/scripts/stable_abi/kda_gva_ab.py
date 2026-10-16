import ctypes,sys,torch,torch_npu
sys.path.insert(0,"/workspace/bartonfang/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/pre_process_fwd_kernel_merged/reference")
import fla_npu.ops.ascendc as A
from fla_npu.ops.ascendc import _aclnn_ctypes as ct
from reference import pre_process_fwd_kernel_merged as ref
T,HK,HV,K,V,BT=256,2,4,128,128,64
def mk(s):
 g=torch.Generator().manual_seed(s)
 k=torch.nn.functional.normalize(torch.randn(T,HK,K,generator=g),dim=-1).bfloat16()
 u=torch.randn(T,HV,V,generator=g).bfloat16()
 be=torch.rand(T,HV,1,generator=g)*0.02
 hk=torch.arange(HV)//(HV//HK)
 w=(be*k[:,hk].float()).bfloat16()
 nb=-(-T//BT)
 gk=(-0.013/BT*(1+torch.rand(nb,BT,HV,K,generator=g)*0.5)).cumsum(1).reshape(-1,HV,K)[:T].contiguous()
 return k,u,w,gk
b=lambda x: x.movedim(1,0).unsqueeze(0).to("npu:0").contiguous()
def direct(kd,wd,ud,gkd):
 hm=ct._zeros((1,HV,K,V+K),kd,dtype=torch.float32)
 ct._call_aclnn("aclnnPreProcessFwdKernelMerged",lambda ctx:[ctx.tensor(kd,"k"),ctx.tensor(wd,"w"),ctx.tensor(ud,"u"),ctx.tensor(None,"g"),ctx.tensor(gkd,"gk"),ctx.tensor(None,"bg"),ctx.tensor(None,"v"),ctx.int_array([0,T]),ctypes.c_int64(BT),ctx.tensor(hm,"hm")],hm)
 torch.npu.synchronize(); return hm.float().cpu()[0].clone()
for r in range(6):
 k,u,w,gk=mk(200+r)
 kd,wd,ud,gkd=b(k),b(w),b(u),b(gk)
 a=A.npu_pre_process_fwd_kernel_merged(kd,wd,ud,gk=gkd,cu_seqlens=[0,T],chunk_size=BT)
 torch.npu.synchronize(); a=a.float().cpu()[0].clone()
 c=direct(kd,wd,ud,gkd)
 want=ref(k,u,w,gk=gk,chunk_size=BT,cu_seqlens=[0,T])
 da=(a-want).abs(); dc=(c-want).abs(); dab=(a-c).abs()
 print(r,"stable/ref h=%.2e m=%.2e | ctypes/ref h=%.2e m=%.2e | stable-vs-ctypes=%.2e"%(da[...,:V].max(),da[...,V:].max(),dc[...,:V].max(),dc[...,V:].max(),dab.max()))