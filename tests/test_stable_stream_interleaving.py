#!/usr/bin/env python3
"""Multi-stream / multi-thread regression for the stable launcher (T4).

vLLM runs several worker threads, each on its own NPU stream, and one decode
step alternates the two operators this launcher serves: the recurrent GDR and
the causal-conv1d update.  A launcher that caches the current stream in a
process-global passes every single-stream test and breaks exactly this mix --
the kernel is enqueued on whichever stream another thread happens to own, so
the ordering between the two operators is lost.  That is the failure mode that
took down the 512-token request: the first visible symptom was an
invalid-address fault in an unrelated kernel, because the stream was already
corrupt by the time it ran.

The probe is the one that caught it: an event pair recorded on the calling
thread's own stream has to bracket the call.  A call that lands on a different
stream leaves those events timing nothing, so ``elapsed_time`` comes back ~0.

``negative_control`` puts the process-global cache back and asserts the probe
now fails, so a green run means the probe still has teeth rather than that it
quietly stopped measuring anything.

usage::

    FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
        python tests/test_stable_stream_interleaving.py [--threads 8] [--rounds 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.ascendc import _stable  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")

# A call that stayed on the calling thread's stream takes far longer than this;
# one that went elsewhere leaves the event pair measuring an empty stream.
MIN_ELAPSED_MS = 0.01


def make_templates(batch: int, nk: int = 8, nv: int = 16, dim: int = 128):
    """Read-only inputs, built once so every thread sees identical values."""

    torch.manual_seed(20260914)
    normalize = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
    return dict(
        query=normalize(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16),
        key=normalize(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16),
        value=torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu"),
        beta=torch.rand(batch, nv, dtype=torch.bfloat16, device="npu"),
        g=torch.rand(batch, nv, dtype=torch.float32, device="npu"),
        scale=dim ** -0.5,
        actual_seq_lengths=torch.tensor([0] + [1] * batch, dtype=torch.int32,
                                        device="npu"),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device="npu"),
        x=(torch.arange(2 * 16, dtype=torch.float32) + 1.0).reshape(2, 16).to(
            torch.bfloat16).npu(),
        weight=(torch.arange(4 * 16, dtype=torch.float32) + 101.0).reshape(
            4, 16).to(torch.bfloat16).npu(),
        bias=(torch.arange(16, dtype=torch.float32) + 201.0).to(
            torch.bfloat16).npu(),
        # Block id 0 is the null block: a sequence that addresses it is skipped
        # and its output row is never written, so the ids start at 1.
        conv_indices=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
        batch=batch,
        nv=nv,
        dim=dim,
    )


def make_paged_state(batch: int, nv: int, dim: int, gap: int = 16384,
                     offset: int = 12288):
    """A non-contiguous state, the spelling the original bug reproduced with."""

    block_stride = nv * dim * dim + gap
    raw = torch.empty((batch + 1) * block_stride * 4, dtype=torch.int8,
                      device="npu")
    state = torch.as_strided(
        raw.view(torch.float32),
        size=(batch + 1, nv, dim, dim),
        stride=(block_stride, dim * dim, dim, 1),
        storage_offset=offset)
    state.zero_()
    return state, raw


def make_case(templates):
    """One iteration's inputs.  The mutable ones are fresh every time."""

    state, raw = make_paged_state(templates["batch"], templates["nv"],
                                  templates["dim"])
    conv_state = (torch.arange(3 * 3 * 16, dtype=torch.float32) + 301.0).reshape(
        3, 3, 16).to(torch.bfloat16).npu()
    return dict(
        templates,
        state=state,
        _state_storage=raw,
        conv_state=conv_state,
        conv_x=templates["x"].clone(),
    )


def run_pair(case):
    """The decode-step mix: recurrent GDR then the conv1d update."""

    recurrent = _stable.npu_recurrent_gated_delta_rule(
        case["query"], case["key"], case["value"], case["state"],
        beta=case["beta"], scale=case["scale"],
        actual_seq_lengths=case["actual_seq_lengths"],
        ssm_state_indices=case["ssm_state_indices"],
        num_accepted_tokens=None, g=case["g"])
    conv = _stable.npu_causal_conv1d_update(
        case["conv_x"], case["conv_state"], case["weight"], case["bias"],
        activation="silu", conv_state_indices=case["conv_indices"])
    return recurrent, conv


def probe(case):
    """(elapsed_ms, recurrent_out, conv_out) for one interleaved pair.

    The events are recorded on whatever stream this thread owns; the call has
    to land there too for the pair to bracket anything.
    """

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    recurrent, conv = run_pair(case)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end), recurrent, conv


def check_negative_control(templates) -> float:
    """Re-install the process-global stream cache; return the probe's elapsed.

    The removed implementation cached one thread's raw stream pointer and
    replayed it for every caller, so a probe running on a freshly created
    stream has its events bracket nothing.
    """

    frozen = int(torch_npu._C._npu_getCurrentRawStream(0))
    original = _stable._raw_stream_fn
    other = torch.npu.Stream()
    try:
        _stable._raw_stream_fn = lambda device_index: frozen
        with torch.npu.stream(other):
            elapsed, _, _ = probe(make_case(templates))
    finally:
        _stable._raw_stream_fn = original
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    torch.npu.set_device(0)
    if not STABLE_LIB:
        print("note: FLA_NPU_STABLE_LIB is unset; relying on the bundled "
              "libfla_npu_stable.so next to the package")
    if not _stable.available():
        raise SystemExit(
            "no Stable-ABI launcher: set FLA_NPU_STABLE_LIB to a built "
            "libfla_npu_stable.so, or install a wheel that bundles one")

    # Built on this thread: the tensors must be identical for every worker, and
    # the device RNG is a single global sequence.
    templates = make_templates(batch=8)
    golden_recurrent, golden_conv = run_pair(make_case(templates))
    torch.npu.synchronize()
    golden = (golden_recurrent, golden_conv)

    errors: list[tuple[int, str]] = []
    notes: list[str] = []
    barrier = threading.Barrier(args.threads + 1, timeout=120)

    def worker(index: int) -> None:
        try:
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                barrier.wait()
                for _ in range(args.rounds):
                    case = make_case(templates)
                    elapsed, recurrent, conv = probe(case)
                    if not elapsed > MIN_ELAPSED_MS:
                        raise AssertionError(
                            f"the pair did not land on this thread's stream "
                            f"(elapsed={elapsed:.4f} ms)")
                    for label, got, want in (("recurrent", recurrent, golden[0]),
                                             ("conv1d", conv, golden[1])):
                        diff = float((got.float() - want.float()).abs().max().item())
                        if diff != 0.0:
                            raise AssertionError(
                                f"{label} parity diff={diff} on a separate "
                                "stream")
                notes.append(f"thread {index}: {args.rounds} pairs ok")
        except Exception as exc:  # noqa: BLE001
            errors.append((index, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(args.threads)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    for note in sorted(notes):
        print(f"  {note}")
    if errors:
        for index, message in errors:
            print(f"FAIL thread {index}: {message}")
        return 1
    print(f"PASS {args.threads} threads x own stream, interleaved "
          f"recurrent+conv1d, parity 0.0")

    elapsed = check_negative_control(templates)
    if elapsed > MIN_ELAPSED_MS:
        print(f"FAIL negative control: the process-global stream cache was "
              f"expected to break the probe, but elapsed={elapsed:.4f} ms "
              f"(> {MIN_ELAPSED_MS} ms) -- the probe no longer detects a "
              "mis-routed call")
        return 1
    print(f"PASS negative control: the global stream cache is detected "
          f"(elapsed={elapsed:.4f} ms)")
    print("ALL PASS: stable multi-stream interleaving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
