"""Private A5 IPC plan for exercising the existing KDA CP forward entry."""

import ctypes as ct
from contextlib import nullcontext
import hashlib
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from fla_npu.ops.ascendc._kcp import KdaCpPlan, KdaCpResult, _signature
from fla_npu.ops.ascendc._kcp_packed import PackedKcpPlan, PackedShard

_WORLD = 8
_HEADS = 32
_BATCH = 16
_SCALE = 128 ** -0.5
_STATUS_WORDS = 2 * _HEADS * 8
_MAILBOX_BYTES = _HEADS * 128 * 128 * 4 + 2 * _STATUS_WORDS * 4


def _check(call, code):
    if code:
        raise RuntimeError(f"{call} failed with ACL status {code}")


def _all_reports(group, local):
    reports = [None] * _WORLD
    dist.all_gather_object(reports, local, group=group)
    return reports


def _raise_rank_errors(reports, phase="setup"):
    errors = [f"rank {rank}: {error}" for rank, (_, error) in enumerate(reports) if error]
    if errors:
        raise RuntimeError(f"IPC plan {phase} failed: " + "; ".join(errors))


def _close_state(group, state):
    guard = torch.npu.device(state.device) if state is not None else nullcontext()
    with guard:
        error = None
        try:
            if state is not None:
                state.stream.synchronize()
        except Exception as exc:
            error = str(exc)
        # A failed drain may leave peer access in flight. Keep mappings intact in that case.
        _raise_rank_errors(_all_reports(group, (None, error)), "drain")

        failed = []
        if state is not None:
            for key in state.imports:
                code = state.acl.aclrtIpcMemClose(key)
                if code:
                    failed.append(str(code))
        _raise_rank_errors(_all_reports(group, (None, ", ".join(failed))), "detach")

        error = None
        try:
            if state is not None and state.key is not None:
                _check("close export", state.acl.aclrtIpcMemClose(state.key))
            if state is not None and state.inbox.value is not None:
                _check("free mailbox", state.acl.aclrtFree(state.inbox))
        except Exception as exc:
            error = str(exc)
        _raise_rank_errors(_all_reports(group, (None, error)), "release")


class _IpcState:
    def __init__(self, library, group, device, rank, tokens, with_zero_bias):
        self.group = group
        self.device = device
        self.rank = rank
        self.tokens = tokens
        self.stream = torch.npu.current_stream(device)
        self.epoch = 0
        self.closed = False
        self.inbox = ct.c_void_p()
        self.key = None
        self.imports = []

        self.acl = ct.CDLL("libascendcl.so")
        pointer, uint64 = ct.c_void_p, ct.c_uint64
        prototypes = {
            "aclrtMalloc": (ct.POINTER(pointer), ct.c_size_t, ct.c_int32),
            "aclrtFree": (pointer,),
            "aclrtMemset": (pointer, ct.c_size_t, ct.c_int32, ct.c_size_t),
            "aclrtDeviceGetBareTgid": (ct.POINTER(ct.c_int32),),
            "aclrtIpcMemGetExportKey": (pointer, ct.c_size_t, ct.c_char_p, ct.c_size_t, uint64),
            "aclrtIpcMemSetImportPid": (ct.c_char_p, ct.POINTER(ct.c_int32), ct.c_size_t),
            "aclrtIpcMemImportByKey": (ct.POINTER(pointer), ct.c_char_p, uint64),
            "aclrtIpcMemClose": (ct.c_char_p,),
        }
        for name, arguments in prototypes.items():
            function = getattr(self.acl, name)
            function.argtypes = arguments
            function.restype = ct.c_int32

        self.library = ct.CDLL(str(library))
        self.launch = self.library.enqueue_kda_cp_ipc
        self.launch.argtypes = (ct.POINTER(pointer),) + (ct.c_uint32,) * 5 + (pointer,)
        self.launch.restype = None
        rank_scratch = getattr(self.library, "kcp_ipc_rank_scratch_bytes", None)
        if rank_scratch is not None:
            rank_scratch.argtypes = (ct.c_uint32, ct.c_uint32)
            rank_scratch.restype = ct.c_uint32
            scratch_bytes = rank_scratch(rank, _WORLD)
        else:
            # The original preview library reported only the maximum scratch size.
            scratch = getattr(self.library, "ipc_chunk32_scratch_bytes", None)
            if scratch is not None:
                scratch.argtypes = ()
                scratch.restype = ct.c_uint32
            scratch_bytes = scratch() if scratch is not None and 4 <= rank <= 6 else 0
        self.scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        self.offsets = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        self.status = torch.zeros((_STATUS_WORDS,), dtype=torch.int32, device=device)
        self.zero_bias = torch.zeros((_HEADS, 128), dtype=torch.float32, device=device) if with_zero_bias else None
        self.stream.synchronize()

    def tgid(self):
        value = ct.c_int32()
        _check("aclrtDeviceGetBareTgid", self.acl.aclrtDeviceGetBareTgid(ct.byref(value)))
        return value.value

    def export(self, peer_pids):
        _check("aclrtMalloc", self.acl.aclrtMalloc(ct.byref(self.inbox), _MAILBOX_BYTES, 3))
        _check("aclrtMemset", self.acl.aclrtMemset(self.inbox, _MAILBOX_BYTES, 0, _MAILBOX_BYTES))
        key = ct.create_string_buffer(65)
        _check("aclrtIpcMemGetExportKey", self.acl.aclrtIpcMemGetExportKey(
            self.inbox, _MAILBOX_BYTES, key, len(key), 0))
        self.key = key.value
        pids = (ct.c_int32 * len(peer_pids))(*peer_pids)
        _check("aclrtIpcMemSetImportPid", self.acl.aclrtIpcMemSetImportPid(self.key, pids, len(pids)))
        return self.key

    def import_peer(self, key):
        if key is None:
            return None
        pointer = ct.c_void_p()
        _check("aclrtIpcMemImportByKey", self.acl.aclrtIpcMemImportByKey(ct.byref(pointer), key, 1))
        self.imports.append(key)
        return pointer

    def close(self):
        if self.closed:
            return
        self.closed = True
        _close_state(self.group, self)

    def forward(self, q, k, v, gate, beta, alpha, bias, output_final_state):
        if self.closed or torch.npu.current_stream(self.device) != self.stream:
            raise RuntimeError("use an open IPC preview plan on its creation stream")
        if self.epoch >= 2**31 - 1:
            raise RuntimeError("IPC preview generation limit reached")
        self.epoch += 1
        output = q.new_empty((1, self.tokens, _HEADS, 128))
        final = gate.new_empty((1, _HEADS, 128, 128)) if output_final_state else None
        tensors = (q, k, v, gate, beta, alpha, bias)
        for tensor in tensors:
            tensor.record_stream(self.stream)
        data = (ct.c_void_p * 15)(
            *(tensor.data_ptr() for tensor in tensors), self.offsets.data_ptr(),
            output.data_ptr(), final.data_ptr() if final is not None else None, self.status.data_ptr(),
            self.inbox, self.following, self.previous, self.scratch.data_ptr())
        with torch.npu.device(self.device):
            self.launch(data, self.tokens, self.rank, _WORLD, self.epoch, output_final_state,
                        self.stream.npu_stream)
        return KdaCpResult(output, final)

    def check_completion(self):
        self.stream.synchronize()
        if self.epoch and not (self.status.cpu()[::8] == self.epoch).all():
            raise RuntimeError(f"rank {self.rank} did not finish IPC generation {self.epoch}")


def create_ipc_preview_plan(q, k, v, g, beta, *, cp_group, A_log, dt_bias, library,
                            output_final_state=False):
    """Build a single-sequence CP8 plan for positive, potentially uneven shards."""
    if (not dist.is_initialized() or dist.get_backend(cp_group) != "gloo"
            or dist.get_world_size(cp_group) != _WORLD):
        raise ValueError("IPC preview requires a CP8 Gloo group")
    rank = dist.get_rank(cp_group)
    signature, errors = _signature(q, k, v, g, beta, A_log, dt_bias, _SCALE,
                                   "none", "fused", _BATCH, output_final_state)
    peer_signature = None
    tokens = None
    if not errors:
        if q.shape[1] != _HEADS or "950" not in torch.npu.get_device_name(q.device):
            errors.append("IPC preview requires A5 and H32")
        if any(format_id not in (0, 2) for format_id in signature[-1]):
            errors.append("IPC preview requires dense storage")
        tokens = q.shape[2]
        if tokens > 2**31 - 1:
            errors.append("IPC preview requires local token counts to fit int32")
        shapes, beta_shape, *options = signature
        # The kernel uses a per-rank T and passes only fixed-size states between ranks.
        peer_signature = (tuple(shape[:2] + shape[3:] for shape in shapes), beta_shape[:2], *options)
    digest = None
    try:
        digest = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(str(exc))
    reports = _all_reports(cp_group, ((peer_signature, socket.gethostname(), digest, q.device.index, tokens), errors))
    _raise_rank_errors(reports)
    if any(report[0][:3] != reports[0][0][:3] for report in reports):
        raise ValueError("IPC preview requires one host, matching tensor metadata and identical kernel binaries")
    if len({report[0][3] for report in reports}) != _WORLD:
        raise ValueError("IPC preview requires eight distinct devices")
    rank_tokens = tuple(report[0][4] for report in reports)

    state = None
    with torch.npu.device(q.device):
        try:
            tgid, error = None, None
            try:
                state = _IpcState(library, cp_group, q.device, rank, tokens, dt_bias is None)
                tgid = state.tgid()
            except Exception as exc:
                error = str(exc)
            _raise_rank_errors(reports := _all_reports(cp_group, (tgid, error)))
            neighbors = tuple(peer for peer in (rank - 1, rank + 1) if 0 <= peer < _WORLD)
            key, error = None, None
            try:
                key = state.export([reports[peer][0] for peer in neighbors])
            except Exception as exc:
                error = str(exc)
            _raise_rank_errors(reports := _all_reports(cp_group, (key, error)))
            error = None
            try:
                state.previous = state.import_peer(reports[rank - 1][0] if rank else None)
                state.following = state.import_peer(reports[rank + 1][0] if rank + 1 < _WORLD else None)
            except Exception as exc:
                error = str(exc)
            _raise_rank_errors(_all_reports(cp_group, (None, error)))
        except Exception:
            if state is not None:
                state.close()
            else:
                _close_state(cp_group, None)
            raise

    shards = tuple(PackedShard((0, count), 0, 0, index + 1 < _WORLD)
                   for index, count in enumerate(rank_tokens))
    rounds = _HEADS // _BATCH
    packed = PackedKcpPlan((0, sum(rank_tokens)), shards, _HEADS, _BATCH, rounds,
                           rounds, rounds, _BATCH)
    return KdaCpPlan(signature, cp_group, rank, _WORLD, None, str(q.device), packed=packed,
                     zero_bias=state.zero_bias, _ipc_preview=state)
