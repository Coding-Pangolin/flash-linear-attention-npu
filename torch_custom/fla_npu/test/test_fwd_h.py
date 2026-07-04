import os
import sys
import logging
import numpy as np
import random
import ml_dtypes
import torch
import torch_npu
from ml_dtypes import bfloat16
from dataclasses import dataclass
import math
# import custom_ops
import fla_npu

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


np.random.seed(1)
torch.manual_seed(1)
if __name__ == "__main__":
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
from typing import Optional


def cdiv(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def _round_elem(x: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    """将张量舍入到 elem_dtype（bf16/fp16）精度，仍在 fp32 容器中计算。"""
    if elem_dtype == torch.float32:
        return x.to(torch.float32)
    return x.to(elem_dtype).to(torch.float32)


def _matmul_npu_aligned(a: torch.Tensor, b: torch.Tensor, elem_dtype: torch.dtype) -> torch.Tensor:
    """bf16/fp16 乘 + fp32 累加，与 Cube MMAD 语义对齐。"""
    return _round_elem(a, elem_dtype) @ _round_elem(b, elem_dtype)


def forward_h_trans_cpu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    keep_fp32: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    golden_mode: str = "fp32",
):
    # 典型场景 HQ=HK=16 HV=32
    # golden_mode:
    #   fp32  - 输入升 fp32，fp32×fp32 矩阵乘（旧同精度标杆）
    #   fp64  - 输入升 fp64，fp64 累加（升精度真值）
    #   npu   - k/w/u 保持 bf16/fp16 乘，fp32 累加；g 用 fp32；状态按 elem_dtype 回写
    dtype_ = k.dtype
    if golden_mode == "fp64":
        compute_dtype = torch.float64
        elem_dtype = None
    elif golden_mode == "npu":
        compute_dtype = torch.float32
        elem_dtype = dtype_
    elif golden_mode != "fp32":
        raise ValueError(f"unsupported golden_mode={golden_mode}")
    else:
        elem_dtype = None

    if keep_fp32 and compute_dtype != torch.float32:
        raise ValueError("keep_fp32=True requires compute_dtype=torch.float32")

    if golden_mode == "npu":
        k = k.to(dtype_)
        w = w.to(dtype_)
        u = u.to(dtype_)
        g = g.float()
    else:
        k = k.to(compute_dtype)
        w = w.to(compute_dtype)
        u = u.to(compute_dtype)
        g = g.to(compute_dtype)

    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = u.shape[1], u.shape[3]

    BT = chunk_size
    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        elif not isinstance(chunk_indices, torch.Tensor):
            chunk_indices = torch.tensor(chunk_indices, dtype=torch.long)
        if chunk_indices.dim() == 1:
            chunk_indices = chunk_indices.view(-1, 2)
    else:
        chunk_indices = None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N = len(cu_seqlens) - 1
        NT = chunk_indices.shape[0]
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    final_state = None

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=compute_dtype)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=compute_dtype)

    head_ratio = HV // HK

    def _chunk_matmul(a, b):
        if elem_dtype is None:
            return a @ b
        return _matmul_npu_aligned(a, b, elem_dtype)

    def _store_elem(x):
        if elem_dtype is None:
            return x
        return _round_elem(x, elem_dtype)

    def _to_compute(x):
        if elem_dtype is None:
            return x.to(compute_dtype)
        return _round_elem(x, elem_dtype)

    def _run_chunk(batch_idx, h_idx, bos, eos, nt_inner, boh):
        for i in range(nt_inner):
            k_sel = torch.zeros((BT, k.shape[-1]), device=k.device, dtype=compute_dtype)
            w_sel = torch.zeros((BT, w.shape[-1]), device=w.device, dtype=compute_dtype)
            u_sel = torch.zeros((BT, u.shape[-1]), device=u.device, dtype=compute_dtype)
            g_sel = torch.zeros((BT), device=g.device, dtype=g.dtype)
            actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

            if cu_seqlens is None:
                k_sel[:actual_len, :] = _to_compute(
                    k[batch_idx, h_idx // head_ratio, bos + i * BT : bos + i * BT + actual_len, :],
                )
                w_sel[:actual_len, :] = _to_compute(
                    w[batch_idx, h_idx, bos + i * BT : bos + i * BT + actual_len, :],
                )
                u_sel[:actual_len, :] = _to_compute(
                    u[batch_idx, h_idx, bos + i * BT : bos + i * BT + actual_len, :],
                )
                g_sel[:actual_len] = g[batch_idx, h_idx, bos + i * BT : bos + i * BT + actual_len]
                state = S[batch_idx, h_idx, boh + i]
            else:
                k_sel[:actual_len, :] = _to_compute(
                    k[0, h_idx // head_ratio, bos + i * BT : bos + i * BT + actual_len, :],
                )
                w_sel[:actual_len, :] = _to_compute(
                    w[0, h_idx, bos + i * BT : bos + i * BT + actual_len, :],
                )
                u_sel[:actual_len, :] = _to_compute(
                    u[0, h_idx, bos + i * BT : bos + i * BT + actual_len, :],
                )
                g_sel[:actual_len] = g[0, h_idx, bos + i * BT : bos + i * BT + actual_len]
                state = S[0, h_idx, boh + i]

            v_new = u_sel - _chunk_matmul(w_sel, state)
            if i != nt_inner - 1:
                g_last = g_sel[actual_len - 1].to(torch.float32)
                g_chunk = g_sel[:actual_len].to(torch.float32)
                v_decay = v_new[:actual_len, :] * torch.exp(g_last - g_chunk).unsqueeze(-1)
                s_decayed = _round_elem(state, elem_dtype) if elem_dtype is not None else state
                s_decayed = s_decayed * torch.exp(g_last)
                s_update = _chunk_matmul(
                    k_sel[:actual_len, :].transpose(-1, -2),
                    v_decay,
                )
                s_next = _store_elem(s_decayed + s_update)
                if cu_seqlens is None:
                    S[batch_idx, h_idx, boh + i + 1] = s_next
                else:
                    S[0, h_idx, boh + i + 1] = s_next

            v_store = _store_elem(v_new[:actual_len, :])
            if cu_seqlens is None:
                v_new_output[batch_idx, h_idx, bos + i * BT : bos + i * BT + actual_len, :] = v_store
            else:
                v_new_output[0, h_idx, bos + i * BT : bos + i * BT + actual_len, :] = v_store

    for n in range(N):
        if cu_seqlens is None: # 定长
            bos = 0
            eos = T
            T_inner = T
            NT_inner = NT
            boh = 0
        else:
            bos = cu_seqlens[n]
            eos = cu_seqlens[n + 1]
            T_inner = eos - bos
            NT_inner = (T_inner + BT - 1) // BT
            boh = chunk_offsets[n]

        for h in range(HV):
            _run_chunk(n if cu_seqlens is None else 0, h, bos, eos, NT_inner, boh)


    #S = S.to(torch.bfloat16)
    #v_new_output = v_new_output.to(torch.bfloat16)
    if keep_fp32:
        return S, v_new_output, None
    S = S.to(dtype_)
    v_new_output = v_new_output.to(dtype_)
    return S, v_new_output, None

class GDNFwdHInput:
    def __init__(self):
        self.batch = int(sys.argv[1])
        self.seqlen = int(sys.argv[2])
        self.k_num_head = int(sys.argv[3])
        self.v_num_head = int(sys.argv[4])
        self.k_head_dim = int(sys.argv[5])
        self.v_head_dim = int(sys.argv[6])
        self.is_varied_len = int(sys.argv[7])
        self.chunk_size = int(sys.argv[8])
        self.use_initial_state = int(sys.argv[9])
        self.store_final_state = int(sys.argv[10])
        self.str_dtype = str(sys.argv[11])
        self.use_actual_input = int(sys.argv[12])
        self.use_actual_output = int(sys.argv[13])
        self.data_path = str(sys.argv[14])

        if self.is_varied_len:
            self.shape_batch = 1
            self.token_batch = self.batch
        else:
            self.shape_batch = self.batch
            self.token_batch = 1
        if self.str_dtype == "half" or self.str_dtype == "fp16" or self.str_dtype == "float16":
            self.dtype = torch.float16
        elif self.str_dtype == "bf16" or self.str_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            logging("[ERROR] dtype must be half or bf16")
            sys.exit()

class GDNFwdHInputTensor:
    def __init__(self, k, w, u, g, cu_seqlens, chunk_indices, initial_state):
        self.k = k
        self.w = w
        self.u = u
        self.g = g
        self.cu_seqlens = cu_seqlens
        self.chunk_indices = chunk_indices
        self.initial_state = initial_state

class GDNFwdHOutputTensor:
    def __init__(self, h, v_new, final_state = None):
        self.h = h
        self.v_new = v_new
        self.final_state = final_state

def parse_actual_input(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    if actual_data.get("chunk_indices") is not None:
        k = actual_data['k'].to(h_input.dtype)
        w = actual_data['w'].to(h_input.dtype)
        u = actual_data['u'].to(h_input.dtype)
        g = actual_data['g'].float()
        cu_raw = actual_data.get('cu_seqlens')
        cu_seqlens = None if cu_raw is None else [int(x) for x in cu_raw]
        chunk_indices = [int(x) for x in actual_data['chunk_indices']]
        initial_state = actual_data.get('initial_state')
        if initial_state is not None:
            initial_state = initial_state.to(h_input.dtype)
        return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_indices, initial_state)

    k = actual_data['k'][:, :, :h_input.k_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    w = actual_data['w'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    u = actual_data['u'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    g = actual_data['g'][:, :, :h_input.v_num_head].transpose(1, 2).contiguous()
    cu_seqlens, chunk_indices = get_cu_offsets(h_input, actual_data.get('cu_seqlens'))
    initial_state = None
    h_input.num_tokens = cu_seqlens[-1] if h_input.is_varied_len else h_input.seqlen
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_indices, initial_state)

def parse_actual_output(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    h = actual_data['h'] if 'h' in actual_data.keys() else actual_data['ref_h']
    v = actual_data['v_new'] if 'v_new' in actual_data.keys() else actual_data['ref_v_new']
    h = h[:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    v = v[:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    return GDNFwdHOutputTensor(h, v)

def gen_seqlen(seqlen, is_varied_len, batch):
    if is_varied_len == 0:
        return None
    cu_seqlens = [0]
    avg_len = seqlen // batch
    for i in range(batch - 1):
        diff = random.randint(avg_len // 2, avg_len * 3 // 2)
        cu_seqlens.append(cu_seqlens[-1] + diff)
    cu_seqlens.append(seqlen)
    return torch.Tensor(cu_seqlens).to(torch.int64)

def gen_wu_data():
    pass

def get_cu_offsets(h_input, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    cu_list = [int(x) for x in cu_seqlens.tolist()]
    chunk_indices = []
    for seq_id in range(len(cu_list) - 1):
        seq_len = cu_list[seq_id + 1] - cu_list[seq_id]
        chunk_num = (seq_len + h_input.chunk_size - 1) // h_input.chunk_size
        for chunk_id in range(chunk_num):
            chunk_indices.extend([seq_id, chunk_id])
    return cu_list, chunk_indices

def gen_decay_data(h_input, cu_seqlens, chunk_offsets):
    base = torch.randint(-15, -5, [h_input.v_num_head])
    bias = torch.empty([h_input.shape_batch, h_input.v_num_head, h_input.seqlen]).uniform_(-2, 0)
    g = base[:, None] + bias

    for shape_batch_idx in range(h_input.shape_batch):
        for v_head_idx in range(h_input.v_num_head):
            for token_batch_idx in range(h_input.token_batch):
                batch_token_start, batch_token_end = cu_seqlens[token_batch_idx], cu_seqlens[token_batch_idx+1]
                batch_tokens = batch_token_end - batch_token_start
                batch_chunks = math.ceil(batch_tokens / h_input.chunk_size)
                for chunk_id in range(batch_chunks):
                    chunk_start_token = batch_token_start + h_input.chunk_size * chunk_id
                    chunk_end_token = min(chunk_start_token + h_input.chunk_size, batch_token_end)
                    g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token] = g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token].cumsum(0)
    return g

def gen_input_data(h_input, rand_wu = True):
    cu_seqlens = gen_seqlen(h_input.seqlen, h_input.is_varied_len, h_input.token_batch)
    cu_seqlens, chunk_indices = get_cu_offsets(h_input, cu_seqlens)
    if rand_wu:
        w = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype)
        u = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen, h_input.v_head_dim], dtype=h_input.dtype)
    else:
        w, u = gen_wu_data()
    k = torch.randn([h_input.shape_batch, h_input.k_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype)
    g = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen], dtype=torch.float)
    # g = gen_decay_data(h_input, cu_seqlens, chunk_offsets)
    if h_input.use_initial_state:
        initial_state = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.token_batch, h_input.k_head_dim, h_input.v_head_dim], dtype=h_input.dtype)
    else:
        initial_state = None
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_indices, initial_state)

def gen_ref_data(h_input, input_tensor):
    h, v, _ = forward_h_trans_cpu(k=input_tensor.k, w=input_tensor.w, u=input_tensor.u, g=input_tensor.g)
    return GDNFwdHOutputTensor(h, v)

def save_data(input_tensor, output_tensor):
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)
    input_tensor.k.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    input_tensor.w.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "w.bin"))
    input_tensor.u.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "u.bin"))
    input_tensor.g.numpy().tofile(os.path.join(WORKSPACE, "data", "g.bin"))
    if input_tensor.cu_seqlens is not None:
        np.array(input_tensor.cu_seqlens).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "cu_seqlens.bin"))

    if input_tensor.initial_state is not None:
        input_tensor.initial_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "initial_state.bin"))

    output_tensor.h.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_ref.bin"))
    output_tensor.v_new.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_ref.bin"))
    if output_tensor.final_state is not None:
        output_tensor.final_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "final_state_ref.bin"))

if __name__ == "__main__":

    gdn_fwd_h_input = GDNFwdHInput()

    if gdn_fwd_h_input.use_actual_input:
        input_tensor = parse_actual_input(gdn_fwd_h_input)
    else:
        input_tensor = gen_input_data(gdn_fwd_h_input)

    if gdn_fwd_h_input.use_actual_output:
        output_tensor = parse_actual_output(gdn_fwd_h_input)
    else:
        output_tensor = gen_ref_data(gdn_fwd_h_input, input_tensor)
    torch.npu.synchronize()

    print("before custom op")
    print(input_tensor.chunk_indices)
    print(input_tensor.cu_seqlens)
    def _as_int_list(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().flatten().tolist()]
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            out = []
            for pair in x:
                out.extend([int(pair[0]), int(pair[1])])
            return out
        return [int(v) for v in x]

    # 与 npu_custom.yaml / FLA chunk_gated_delta_rule_fwd_h 对齐：k,w,u 位置参数；g 及之后为关键字（g 当前不可为 None）
    result = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        input_tensor.k.npu(),
        input_tensor.w.npu(),
        input_tensor.u.npu(),
        g=input_tensor.g.npu(),
        initial_state=(
            input_tensor.initial_state.npu()
            if input_tensor.initial_state is not None
            else None
        ),
        output_final_state=bool(gdn_fwd_h_input.store_final_state),
        chunk_size=gdn_fwd_h_input.chunk_size,
        cu_seqlens=_as_int_list(input_tensor.cu_seqlens),
        chunk_indices=_as_int_list(input_tensor.chunk_indices),
    )
    print("after custom op")
    torch_npu._C._npu_synchronize()
    print("after synchronize")
    save_data(input_tensor, output_tensor)
    result[0].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_npu.bin"))
    result[1].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_npu.bin"))

    print("Done")