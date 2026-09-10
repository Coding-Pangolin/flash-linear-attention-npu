"""Stable-ABI 竖切验证（Phase 1）：recurrent GDR 走 torch.ops 注册的 stable 实现。

用法（221，已 build 出 libfla_npu_thin.so）：
    FLA_NPU_STABLE_LIB=/path/libfla_npu_thin.so PYTHONPATH=<env> \
        python tests/regression_stable_abi.py

覆盖计划里的 T1（parity）、T2（mutation 契约）、T3（多线程多 stream）、
T5（host A/B）、T8（stable stream 探针）。T6（ELF 符号审计）由
tools/stable_abi_audit.py --lib 单独跑。
"""
from __future__ import annotations

import os
import threading
import time

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _aclnn_ctypes as ct  # noqa: E402
from fla_npu.ops.ascendc import _stable, _wrap_mutable_direct_op  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")


def pct(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bench(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.npu.synchronize()
    return pct(ts, 0.5)


def make_inputs(batch=8, nk=8, nv=16, dim=128, gap=16384, offset=12288):
    block_stride = nv * dim * dim + gap

    def make_state():
        raw = torch.empty((batch + 1) * block_stride * 4, dtype=torch.int8,
                          device="npu")
        state = torch.as_strided(
            raw.view(torch.float32),
            size=(batch + 1, nv, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset)
        state.zero_()
        return state, raw

    norm = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
    query = norm(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16)
    key = norm(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16)
    value = torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(batch, nv, dtype=torch.bfloat16, device="npu")
    g = torch.rand(batch, nv, dtype=torch.float32, device="npu")
    asl = torch.tensor([0] + [1] * batch, dtype=torch.int32, device="npu")
    ssi = torch.arange(batch, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    return dict(query=query, key=key, value=value, beta=beta, g=g,
                scale=dim ** -0.5, actual_seq_lengths=asl,
                ssm_state_indices=ssi, make_state=make_state)


def call_public_stable(inputs, state):
    return _stable.npu_recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state,
        beta=inputs["beta"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None, g=inputs["g"])


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20260911)
    if not STABLE_LIB:
        raise SystemExit("set FLA_NPU_STABLE_LIB to the built libfla_npu_thin.so")
    assert _stable.available(), "stable library loaded but op not registered"
    print(f"loaded {STABLE_LIB}")
    inputs = make_inputs()

    # --- T1 parity vs ctypes (non-contiguous paged state) --------------------
    state_c, _ = inputs["make_state"]()
    state_s, _ = inputs["make_state"]()
    out_c = ct.npu_recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state_c,
        beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None)
    out_s = call_public_stable(inputs, state_s)
    torch.npu.synchronize()
    diff_out = float((out_c.float() - out_s.float()).abs().max().item())
    diff_state = float((state_c.float() - state_s.float()).abs().max().item())
    assert diff_out == 0.0 and diff_state == 0.0, (
        f"T1 diff out={diff_out} state={diff_state}")
    print(f"PASS T1 parity vs ctypes (out={diff_out}, state={diff_state})")

    # --- T2 mutation contract -------------------------------------------------
    # (a) dispatcher alone: schema alias does NOT bump the version counter.
    state_raw, _ = inputs["make_state"]()
    before = int(state_raw._version)
    call_public_stable(inputs, state_raw)
    torch.npu.synchronize()
    raw_bumped = int(state_raw._version) > before
    print(f"T2a raw dispatcher op: version {'bumped' if raw_bumped else 'NOT bumped'}"
          f"  (documented gap: contract stays in the Python layer)")
    # (b) with the shared mutation contract applied (production shape).
    stable_op = _wrap_mutable_direct_op(
        "npu_recurrent_gated_delta_rule", _stable.npu_recurrent_gated_delta_rule)
    state_w, _ = inputs["make_state"]()
    before = int(state_w._version)
    stable_op(inputs["query"], inputs["key"], inputs["value"], state_w,
              beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
              actual_seq_lengths=inputs["actual_seq_lengths"],
              ssm_state_indices=inputs["ssm_state_indices"],
              num_accepted_tokens=None)
    torch.npu.synchronize()
    after = int(state_w._version)
    assert after == before + 1, f"T2b version {before} -> {after}"
    grad_state, _ = inputs["make_state"]()
    grad_state = grad_state.detach().requires_grad_(True)
    try:
        stable_op(inputs["query"], inputs["key"], inputs["value"], grad_state,
                  beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
                  actual_seq_lengths=inputs["actual_seq_lengths"],
                  ssm_state_indices=inputs["ssm_state_indices"],
                  num_accepted_tokens=None)
    except RuntimeError as exc:
        assert "must not require gradients" in str(exc), str(exc)
        print(f"PASS T2 mutation contract (version +1, requires_grad rejected)")
    else:
        raise AssertionError("T2b: requires_grad state was accepted")

    # --- T8 stable stream probe ----------------------------------------------
    device_index = int(torch.npu.current_device())
    py_raw = int(torch_npu._C._npu_getCurrentRawStream(device_index))
    shim_raw, shim_id = _stable.stream_probe(device_index)
    print(f"T8 stream probe: python_raw={py_raw} shim_raw={shim_raw} "
          f"stable_id={shim_id} "
          f"({'MATCH' if shim_raw == py_raw else 'MISMATCH'})")

    # --- T3 multi-thread / multi-stream -------------------------------------
    golden = None
    state_g, _ = inputs["make_state"]()
    golden = ct.npu_recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state_g,
        beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None).clone()
    torch.npu.synchronize()
    n_threads = 4
    barrier = threading.Barrier(n_threads)
    errors: list[tuple[int, str]] = []

    def worker(index):
        try:
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                barrier.wait(timeout=60)
                for _ in range(3):
                    state, _ = inputs["make_state"]()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    out = call_public_stable(inputs, state)
                    end.record()
                    torch.npu.synchronize()
                    elapsed = start.elapsed_time(end)
                    if not elapsed > 0.01:
                        raise AssertionError(
                            f"thread {index}: op did not land on its own stream "
                            f"(elapsed={elapsed:.4f} ms)")
                    diff = float((out.float() - golden.float()).abs().max().item())
                    if diff != 0.0:
                        raise AssertionError(f"thread {index}: parity diff={diff}")
        except Exception as exc:  # noqa: BLE001
            errors.append((index, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, f"T3 failures: {errors}"
    print(f"PASS T3 {n_threads} threads x own stream "
          f"(events landed on the calling stream, parity 0.0)")

    # --- T5 host A/B ---------------------------------------------------------
    from fla_npu.ops.ascendc import _thin

    state_a, _ = inputs["make_state"]()
    state_b, _ = inputs["make_state"]()
    with torch.no_grad():
        a = bench(lambda: ct.npu_recurrent_gated_delta_rule(
            inputs["query"], inputs["key"], inputs["value"], state_a,
            beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=None))
        b = bench(lambda: _thin.npu_recurrent_gated_delta_rule(
            inputs["query"], inputs["key"], inputs["value"], state_b,
            beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=None))
        c = bench(lambda: call_public_stable(inputs, state_s))
        d = bench(lambda: stable_op(
            inputs["query"], inputs["key"], inputs["value"], state_w,
            beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=None))
    print(f"T5 host P50  ctypes={a:.4f}  pybind-thin={b:.4f}  "
          f"stable={c:.4f}  stable+contract={d:.4f} ms")
    print("ALL PASS: stable-abi Phase 1")


if __name__ == "__main__":
    main()
