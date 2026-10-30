"""Parallel scenario evaluation harness for the Smart Buildings competition.

Runs one full episode per scenario, each in its own process (gin config is
process-global), with a plain reset/step loop that matches the Kaggle gateway.

Usage (on the VM, inside /data/smartbuild/.venv):

    python -m sb.eval.harness --policy smart_buildings.baselines.pid_plus:PIDWaterResetPolicy \
        --name pid_water --workers 30

    # our policy with a parameter file
    python -m sb.eval.harness --policy sb.policy:MyPolicy --params params/best.json --name v1

Score = mean over scenarios of |mean per-step reward| (the leaderboard metric).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS/OpenMP thread per worker; we parallelize across scenarios instead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

KIT = Path(os.environ.get("SB_KIT", "/data/smartbuild/starter-kit"))
DEFAULT_SCENARIOS = KIT / "smart_buildings" / "training_scenarios"
RUNS = Path(os.environ.get("SB_RUNS", "/data/smartbuild/runs"))


def load_class(spec: str):
    module_name, class_name = spec.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def discover(root: Path, include: list[str] | None) -> list[Path]:
    dirs = sorted(p.parent for p in root.glob("*/*/config.gin"))
    if include:
        dirs = [d for d in dirs if any(f"{d.parent.name}/{d.name}".startswith(s) or d.parent.name == s for s in include)]
    return dirs


def run_scenario(scenario_dir: str, policy_spec: str, params: dict | None, steps: int, occupancy_seed: int | None) -> dict:
    """Run a single episode. Executed in a worker process."""
    import numpy as np
    import gin
    from smart_buildings import evaluate as ev

    scenario_dir = Path(scenario_dir).resolve()
    os.chdir(KIT)  # config.gin paths are relative to the starter-kit root
    gin.clear_config()
    with gin.unlock_config():
        gin.parse_config_file(str(scenario_dir / "config.gin"))
        if occupancy_seed is not None:
            gin.bind_parameter("RandomizedArrivalDepartureOccupancy.seed", occupancy_seed)
    from smart_buildings.smart_control.environment import hybrid_action_environment

    env = hybrid_action_environment.HybridActionEnvironment()
    meta = ev.create_building_meta_data(env, scenario_dir)
    spec = env.action_spec()

    # Capture the reward breakdown of each step.
    captured = {}
    orig = env.reward_function.compute_reward

    def wrapped(info):
        resp = orig(info)
        captured["r"] = resp
        return resp

    env.reward_function.compute_reward = wrapped

    cls = load_class(policy_spec)
    policy = cls(meta, spec, params) if params is not None else ev.initialize_policy(cls, meta, spec)

    obs, _ = env.reset()
    total = comfort = energy = carbon = 0.0
    policy_time = 0.0
    n = 0
    worst = 0.0
    t_start = time.time()
    for _ in range(steps):
        t0 = time.perf_counter()
        action = ev.validate_action(policy.get_action(obs), spec)
        policy_time += time.perf_counter() - t0
        obs, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if not np.isfinite(reward):
            raise RuntimeError(f"non-finite reward at step {n}: {reward}")
        total += reward
        worst = min(worst, reward)
        r = captured.get("r")
        if r is not None:
            comfort += r.normalized_productivity_regret
            energy += r.normalized_energy_cost
            carbon += r.normalized_carbon_emission
        n += 1
        if terminated or truncated:
            break
    # Weighted components (sum to total): 0.6*comfort - 0.2*energy - 0.2*carbon
    return {
        "scenario": f"{scenario_dir.parent.name}/{scenario_dir.name}",
        "steps": n,
        "total_reward": total,
        "mean_reward": total / max(n, 1),
        "comfort_part": 0.6 * comfort / max(n, 1),
        "energy_part": -0.2 * energy / max(n, 1),
        "carbon_part": -0.2 * carbon / max(n, 1),
        "worst_step": worst,
        "policy_ms_per_step": 1000 * policy_time / max(n, 1),
        "wall_s": time.time() - t_start,
    }


def evaluate(scenarios: list[Path], policy_spec: str, params: dict | None, steps: int,
             workers: int, occupancy_seed: int | None = None, verbose: bool = True) -> dict:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_scenario, str(s), policy_spec, params, steps, occupancy_seed): s for s in scenarios}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:  # report and keep going
                res = {"scenario": str(futs[fut]), "error": repr(exc)}
            results.append(res)
            if verbose:
                if "error" in res:
                    print(f"  FAIL {res['scenario']}: {res['error']}", flush=True)
                else:
                    print(f"  {res['scenario']:40s} mean={res['mean_reward']:.5f} "
                          f"(comfort {res['comfort_part']:.5f} energy {res['energy_part']:.5f} "
                          f"carbon {res['carbon_part']:.5f}) {res['wall_s']:.0f}s "
                          f"{res['policy_ms_per_step']:.2f}ms/step", flush=True)
    ok = [r for r in results if "error" not in r]
    results.sort(key=lambda r: r["scenario"])
    score = sum(abs(r["mean_reward"]) for r in ok) / max(len(ok), 1)
    return {"score": score, "n_ok": len(ok), "n_total": len(results), "results": results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="module:Class")
    p.add_argument("--params", help="JSON file with policy parameters (passed as 3rd ctor arg)")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS), help="root dir with <building>/<weather>/config.gin")
    p.add_argument("--include", nargs="*", help="building names or building/weather prefixes")
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--occupancy-seed", type=int)
    p.add_argument("--name", default=None, help="run name; results go to runs/<name>.json")
    args = p.parse_args(argv)

    scenarios = discover(Path(args.scenarios), args.include)
    params = json.loads(Path(args.params).read_text()) if args.params else None
    print(f"Evaluating {args.policy} on {len(scenarios)} scenarios, {args.steps} steps, {args.workers} workers", flush=True)
    t0 = time.time()
    out = evaluate(scenarios, args.policy, params, args.steps, args.workers, args.occupancy_seed)
    out.update({"policy": args.policy, "params": params, "steps": args.steps,
                "scenarios_root": args.scenarios, "occupancy_seed": args.occupancy_seed,
                "elapsed_s": time.time() - t0})
    print(f"\nSCORE {out['score']:.6f}  ({out['n_ok']}/{out['n_total']} ok, {out['elapsed_s']:.0f}s)")
    if args.name:
        RUNS.mkdir(parents=True, exist_ok=True)
        path = RUNS / f"{args.name}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["n_ok"] else 1)
