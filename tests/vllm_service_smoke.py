#!/usr/bin/env python3
"""Service-level smoke test for a running vLLM-Ascend server (T7).

The per-operator parity suite compares two host paths with the same kernel, so
it cannot see the failure modes that only appear in a server: several worker
threads on several NPU streams, an asynchronous engine, and an operator mix
where recurrent and conv1d calls interleave.  That gap is not hypothetical --
the earlier thin launcher cached the current stream in a process-global and
only survived single-stream use.

What this checks, against a server that is already running:
  1. a single 512-token request completes (the earlier crash showed up around
     token 257);
  2. repeating it N times stays healthy (catches state corruption that only
     shows up after the first request);
  3. concurrent requests at 8 / 32 / 64 keep returning well-formed answers;
  4. the service is still serving afterwards (a request after the load).

Usage::

    python tests/vllm_service_smoke.py --url http://127.0.0.1:8000 \\
        --model /path/to/model --tokens 512 --repeats 20 \\
        --concurrency 8 32 64 --json report.json

Exit code is non-zero when any phase failed, so it can gate a release.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


DEFAULT_PROMPT = "Repeat the alphabet, then explain what a linear attention kernel does."


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _printable(text: str) -> bool:
    """A garbled kernel usually returns control bytes or replacement chars."""

    if not text.strip():
        return False
    allowed = set(string.printable)
    return all(character in allowed for character in text)


def complete(base: str, model: str, tokens: int, timeout: float) -> dict:
    """One completion; raises on transport failure."""

    started = time.perf_counter()
    body = _post(base.rstrip("/") + "/v1/completions",
                 {"model": model, "prompt": DEFAULT_PROMPT,
                  "max_tokens": tokens, "temperature": 0.0},
                 timeout)
    elapsed = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0]
    text = choice.get("text", "")
    return {"seconds": elapsed, "text_length": len(text),
            "finish_reason": choice.get("finish_reason"),
            "printable": _printable(text),
            "usage": body.get("usage")}


def health(base: str, timeout: float = 30.0) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health",
                                    timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def single_run(base: str, model: str, tokens: int, repeats: int,
               timeout: float, report: dict) -> bool:
    print(f"== single request x{repeats} ({tokens} tokens)")
    ok = True
    for index in range(repeats):
        try:
            result = complete(base, model, tokens, timeout)
        except Exception as exc:
            print(f"  run {index + 1}: FAILED {type(exc).__name__}: {exc}")
            report["single"].append({"run": index + 1, "error": str(exc)})
            ok = False
            break
        verdict = ("ok" if result["printable"] and result["finish_reason"]
                   in ("stop", "length") else "suspicious")
        print(f"  run {index + 1}: {verdict} in {result['seconds']:.1f}s "
              f"({result['text_length']} chars, finish={result['finish_reason']})")
        report["single"].append(result)
        if verdict != "ok":
            ok = False
    return ok


def concurrent_run(base: str, model: str, tokens: int, concurrency: int,
                   timeout: float, report: dict) -> bool:
    print(f"== concurrency {concurrency}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(complete, base, model, tokens, timeout)
                   for _ in range(concurrency)]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"error": f"{type(exc).__name__}: {exc}"})
    elapsed = time.perf_counter() - started
    failures = [row for row in results if "error" in row]
    suspicious = [row for row in results
                  if "error" not in row
                  and not (row["printable"]
                           and row["finish_reason"] in ("stop", "length"))]
    print(f"  {concurrency - len(failures)}/{concurrency} ok in "
          f"{elapsed:.1f}s ({len(suspicious)} suspicious)")
    for row in failures[:3]:
        print(f"    failed: {row['error']}")
    report["concurrency"].append({"concurrency": concurrency,
                                  "seconds": elapsed,
                                  "failures": failures,
                                  "suspicious": suspicious})
    return not failures and not suspicious


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="*", default=[8, 32, 64])
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    report: dict = {"url": args.url, "model": args.model,
                    "tokens": args.tokens, "single": [], "concurrency": []}
    if not health(args.url):
        print(f"FAIL: {args.url}/health is not answering")
        return 2

    ok = single_run(args.url, args.model, args.tokens, args.repeats,
                    args.timeout, report)
    for concurrency in args.concurrency:
        ok = concurrent_run(args.url, args.model, args.tokens, concurrency,
                            args.timeout, report) and ok

    print("== final health check (a crashed worker leaves the service down)")
    still_alive = health(args.url)
    report["healthy_after"] = still_alive
    if not still_alive:
        ok = False

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"report -> {args.json}")
    print("PASS: service stable" if ok else "FAIL: see the phases above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
