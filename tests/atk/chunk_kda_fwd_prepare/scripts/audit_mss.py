"""Audit raw MSS severity and coverage without equating warnings with bugs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from delivery_cases import load, write_json
from summarize_run import summarize


def inspect_log(path, expected_keys, tool):
    starts, finishes, clean = Counter(), Counter(), Counter()
    warning_classes, error_classes = Counter(), Counter()
    key_warnings, key_errors = Counter(), Counter()
    samples, error_samples = [], []
    warnings = errors = 0
    tools = set()
    active_key = None
    pending_bounds = False
    log_hash = hashlib.sha256()
    # Stream the raw log: target memcheck logs may be much larger than memory.
    with Path(path).open("rb") as stream:
        for raw in stream:
            log_hash.update(raw)
            line = raw.decode("utf-8", errors="replace")
            start = (re.search(r"Start (\w+) sanitizer on kernel (\S+)", line)
                     if "sanitizer on kernel" in line else None)
            finish = (re.search(r"Sanitizer finished on kernel (\S+?)\. (.*)", line)
                      if "Sanitizer finished on kernel" in line else None)
            if start:
                tools.add(start[1])
                if start[2].startswith("ChunkKdaFwdPrepare_"):
                    starts[start[2]] += 1
                    active_key = start[2].rsplit("_", 1)[1]
                else:
                    active_key = "other_kernel"
            lower = line.lower()
            if pending_bounds:
                if "on gm" in lower:
                    warning_classes["out_of_bounds_warning"] -= 1
                    if not warning_classes["out_of_bounds_warning"]:
                        del warning_classes["out_of_bounds_warning"]
                    warning_classes["gm_ownership"] += 1
                pending_bounds = False
            warning = "warning" in lower and bool(re.search(r"\bwarning\b", lower))
            error = (("error" in lower or "fatal" in lower)
                     and bool(re.search(r"\b(?:error|fatal)\s*:", lower)))
            if warning or error:
                warnings += warning
                errors += error
                # These labels describe tool messages, not adjudicated defects.
                text = lower
                if warning:
                    category = ("gm_ownership" if "out of bounds" in text and "on gm" in text
                                else "out_of_bounds_warning" if "out of bounds" in text
                                else "redundant_wait" if "redundant wait" in text
                                else "uninstrumented_kernel" if "not instrument" in text
                                else "register_state" if any(x in text for x in (
                                    "ffts_base_addr", "vector_mask", "ctrl["))
                                else "other_warning")
                    warning_classes[category] += 1
                    pending_bounds = category == "out_of_bounds_warning"
                    key_warnings[active_key or "unattributed"] += 1
                if error:
                    category = "potential_race" if "race" in text or re.search(r"\b(?:RAW|WAR|WAW)\b", line) else "other_error"
                    error_classes[category] += 1
                    key_errors[active_key or "unattributed"] += 1
                    if len(error_samples) < 10:
                        error_samples.append(dict(key=active_key, text=line.strip()[:1000]))
                if len(samples) < 20:
                    samples.append(dict(key=active_key, severity="ERROR" if error else "WARNING",
                                        text=line.strip()[:1000]))
            if finish and finish[1].startswith("ChunkKdaFwdPrepare_"):
                finishes[finish[1]] += 1
                if finish[2].strip() == "No error detected.":
                    clean[finish[1]] += 1
                active_key = None
    observed = sorted({int(name.rsplit("_", 1)[1]) for name in starts})
    key_counts = Counter()
    for name, count in starts.items():
        key_counts[int(name.rsplit("_", 1)[1])] += count
    complete = (observed == sorted(expected_keys) and bool(starts)
                and starts == finishes and tools == {tool}
                and all(count == 1 for count in key_counts.values()))
    clean_complete = complete and starts == clean
    review = ("instrumentation_incomplete" if not complete else "error_review_required" if errors
              else "warning_review_required" if warnings else "unexplained_nonclean_finish"
              if not clean_complete else "clean")
    return dict(expected_keys=sorted(expected_keys), observed_keys=observed,
                starts=dict(starts), finishes=dict(finishes), clean_finishes=dict(clean),
                tools=sorted(tools), warning_lines=warnings, error_lines=errors,
                warning_classes=dict(warning_classes), error_classes=dict(error_classes),
                warnings_by_key=dict(key_warnings), errors_by_key=dict(key_errors),
                diagnostic_samples=samples, error_samples=error_samples,
                log_sha256=log_hash.hexdigest(),
                complete_instrumentation=complete, complete_clean_kernels=clean_complete,
                review_state=review,
                raw_log_clean=clean_complete and warnings == errors == 0)


def audit(run, manifest, tool):
    run, manifest = Path(run), Path(manifest)
    identity = load(run/"run.json")
    package = load(manifest)
    if identity["scope"] != "mssanitizer":
        raise ValueError("An MSS run is required")
    if identity["package_sha256"] != package["package_sha256"]:
        raise ValueError("Run and instrumented package identities differ")
    native = summarize(run)
    result = inspect_log(run/"sanitizer.log", package["actual_metadata_key_set"], tool)
    result.update(native_all_passed=native["native_all_passed"],
                  source_commit=identity["source_commit"],
                  package_sha256=identity["package_sha256"], tool=tool,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  acceptance="not_accepted")
    if (result["complete_instrumentation"] and not result["error_lines"]
            and result["warning_lines"] and native["native_all_passed"]):
        result["acceptance"] = "warnings_pending_review"
    if result["raw_log_clean"] and native["native_all_passed"]:
        result["acceptance"] = "instrumented_native_pass"
    write_json(run/"mss_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Exact instrumented build manifest, including real metadata keys")
    parser.add_argument("--tool", choices=("memcheck", "racecheck", "initcheck", "synccheck"), required=True)
    args = parser.parse_args()
    result = audit(args.run, args.manifest, args.tool)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["acceptance"] == "instrumented_native_pass" else 1)
