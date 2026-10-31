"""Compare two harness runs scenario by scenario, worst first.

    python scripts/compare_runs.py runs/wide_holdout_g22.json runs/wide_holdout_defaults.json
"""

import json
import sys


def load(path: str) -> dict:
    return {x["scenario"]: x for x in json.load(open(path))["results"] if "error" not in x}


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(f"{'scenario':34s} {'A':>8s} {'comfort':>8s} {'energy':>8s} {'carbon':>8s} | {'B':>8s}")
    for s in sorted(a, key=lambda s: a[s]["mean_reward"]):
        x = a[s]
        other = f"{-b[s]['mean_reward']:8.5f}" if s in b else "       -"
        print(f"{s:34s} {-x['mean_reward']:8.5f} {-x['comfort_part']:8.5f} {-x['energy_part']:8.5f} "
              f"{-x['carbon_part']:8.5f} | {other}")
    common = [s for s in a if s in b]
    sa = sum(-a[s]["mean_reward"] for s in common) / len(common)
    sb = sum(-b[s]["mean_reward"] for s in common) / len(common)
    print(f"mean over {len(common)} common scenarios: A {sa:.5f}  B {sb:.5f}")


if __name__ == "__main__":
    main()
