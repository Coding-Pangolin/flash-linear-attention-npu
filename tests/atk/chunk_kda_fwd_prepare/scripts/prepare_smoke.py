"""Select small real generic cases covering output modes/layouts/auxiliary dtypes."""
import math
from delivery_cases import ROOT, OP, attr, inputs, load, write_json


def features(case):
    i = inputs(case)
    return {("mode", attr(case, "backward_mode")), ("layout", attr(case, "layout")),
            ("gate_dtype", i["g"]["dtype"]), ("beta_dtype", i["beta"]["dtype"]),
            ("gate", attr(case, "use_gate_in_kernel"), attr(case, "safe_gate"))}


def select():
    cases = load(ROOT / f"atk_{OP}.json")
    small = sorted(cases, key=lambda c: sum(math.prod(x.get("shape", []))
        for x in inputs(c).values() if x["type"] == "tensor"))
    wanted = set().union(*(features(c) for c in cases))
    picked = []
    for case in small:
        useful = wanted & features(case)
        if useful:
            picked.append(case)
            wanted -= useful
        if not wanted:
            break
    if wanted:
        raise ValueError(f"Missing smoke features: {wanted}")
    return picked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = select()
    write_json(args.out, cases)
    print(f"smoke logical cases={len(cases)}; original seed retained for each case")
