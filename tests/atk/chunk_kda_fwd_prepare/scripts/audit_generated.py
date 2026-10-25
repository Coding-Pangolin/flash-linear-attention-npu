"""Compare native ATK-generated case inputs and standard against frozen configs."""
import argparse
import json
from delivery_cases import ROOT, OP, STANDARD, load


def audit(path):
    original = {c["name"]: c for c in load(ROOT / f"atk_{OP}.json")}
    generated = load(path)
    if len(generated) != 200 or {c["name"] for c in generated} != set(original):
        raise ValueError("Generated suite changed case count/names")
    for case in generated:
        if case["inputs"] != original[case["name"]]["inputs"]:
            raise ValueError(f"Generated input changed: {case['name']}")
        if case["standard"]["acc"] != STANDARD["acc"]:
            raise ValueError(f"Generated standard changed: {case['name']}")
    return {"native_generated_cases": 200, "input_and_standard_match": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(audit(args.path), sort_keys=True))
