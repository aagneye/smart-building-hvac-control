"""Tune controller parameters with CMA-ES on a set of scenarios.

Every (candidate, scenario) pair of a generation runs in one shared process
pool, so a generation costs about one episode of wall time per wave.
Parameters are searched in a [0, 1] box mapped onto the bounds in params.SPEC.

    python -m sb.eval.tune_cmaes --name tune1 --steps 2016 --popsize 12 --workers 60 \
        --train building_train_1 building_train_2 building_train_4 building_train_5 \
        --holdout building_train_7 building_train_8 --generations 40

Writes runs/<name>/best.json (best params so far), history.csv and holdout.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sb import params as P
from sb.eval.harness import DEFAULT_SCENARIOS, RUNS, discover, run_scenario

POLICY = "sb.policy:ControllerV1"


def to_params(x: np.ndarray, names: list[str], base: dict) -> dict:
    p = dict(base)
    for v, n in zip(np.clip(x, 0.0, 1.0), names):
        lo, hi = P.SPEC[n][1], P.SPEC[n][2]
        p[n] = float(lo + v * (hi - lo))
    return p


def to_unit(p: dict, names: list[str]) -> np.ndarray:
    return np.array([(p[n] - P.SPEC[n][1]) / (P.SPEC[n][2] - P.SPEC[n][1]) for n in names])


def score_candidates(pool, cands: list[dict], scenarios: list[Path], steps: int) -> list[float]:
    futs = [[pool.submit(run_scenario, str(s), POLICY, c, steps, None) for s in scenarios] for c in cands]
    scores = []
    for row in futs:
        vals = []
        for f in row:
            try:
                vals.append(abs(f.result()["mean_reward"]))
            except Exception:  # a crash counts as a very bad score
                vals.append(1.0)
        scores.append(float(np.mean(vals)))
    return scores


def main(argv=None):
    import cma

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="building names or building/weather prefixes")
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--start", help="JSON params to start from (defaults otherwise)")
    ap.add_argument("--only", nargs="*", help="tune only these parameter names")
    ap.add_argument("--holdout-every", type=int, default=5)
    args = ap.parse_args(argv)

    root = Path(args.scenarios)
    train = discover(root, args.train)
    holdout = discover(root, args.holdout) if args.holdout else []
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    base = P.merged(json.loads(Path(args.start).read_text()) if args.start else None)
    names = args.only or P.NAMES
    x0 = to_unit(base, names)
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"bounds": [0, 1], "popsize": args.popsize, "seed": 1, "verbose": -9})
    print(f"tuning {len(names)} params on {len(train)} scenarios ({args.steps} steps), holdout {len(holdout)}", flush=True)

    best = (np.inf, base)
    hist = open(out / "history.csv", "a")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        base_score = score_candidates(pool, [base], train, args.steps)[0]
        print(f"start score {base_score:.6f}", flush=True)
        best = (base_score, base)
        for gen in range(1, args.generations + 1):
            t0 = time.time()
            xs = es.ask()
            cands = [to_params(x, names, base) for x in xs]
            scores = score_candidates(pool, cands, train, args.steps)
            es.tell(xs, scores)
            i = int(np.argmin(scores))
            if scores[i] < best[0]:
                best = (scores[i], cands[i])
                (out / "best.json").write_text(json.dumps({k: best[1][k] for k in names} | {"_score": best[0]}, indent=1))
            hist.write(f"{gen},{min(scores):.6f},{np.median(scores):.6f},{best[0]:.6f},{time.time() - t0:.0f}\n")
            hist.flush()
            print(f"gen {gen:3d} min {min(scores):.6f} median {np.median(scores):.6f} best {best[0]:.6f} ({time.time() - t0:.0f}s)", flush=True)
            if holdout and gen % args.holdout_every == 0:
                h = score_candidates(pool, [best[1]], holdout, args.steps)[0]
                with open(out / "holdout.csv", "a") as f:
                    f.write(f"{gen},{best[0]:.6f},{h:.6f}\n")
                print(f"  holdout {h:.6f} (train {best[0]:.6f})", flush=True)
    print(f"best {best[0]:.6f} -> {out / 'best.json'}")


if __name__ == "__main__":
    main()
