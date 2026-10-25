"""Generate four Forward supplements, retaining every original200 Save output."""
from delivery_cases import ROOT, OP, forward_cases, write_json

if __name__ == "__main__":
    cases = forward_cases()
    write_json(ROOT/f"atk_{OP}_forward.json", cases)
    print(f"Generated {len(cases)} Forward supplements; original200 unchanged")
