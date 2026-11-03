"""Where does a policy's loss come from? Splits reward and energy by period and equipment.

    python scripts/energy_breakdown.py --policy sb.policy_v2:ControllerV2 \
        --params params/v2_tuned_g22.json --include building_train_1/fairbanks_1 --name eb_v2

Periods, judged from the observation at the start of each step:
  occupied       people in the building
  empty_workday  Monday-Friday with nobody in (nights, early mornings, evenings, holidays)
  weekend        Saturday and Sunday
Equipment energy comes from the simulator's reward info: fans, chiller
electricity, pump, boiler gas (kWh). Reward parts are mean per step over the
whole episode, so the period parts add up to the scenario's score.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sb.eval.harness import DEFAULT_SCENARIOS, KIT, RUNS, discover, load_class  # noqa: E402

PERIODS = ("occupied", "empty_workday", "weekend")
PARTS = ("comfort", "energy", "carbon")
EQUIP = ("fan_kWh", "chiller_kWh", "pump_kWh", "gas_kWh")


def run(scenario_dir: str, policy_spec: str, params: dict | None, steps: int) -> dict:
    import gin
    import numpy as np
    from smart_buildings import evaluate as ev
    from sb.obs import build_layout, decode

    d = Path(scenario_dir).resolve()
    os.chdir(KIT)
    gin.clear_config()
    with gin.unlock_config():
        gin.parse_config_file(str(d / "config.gin"))
    from smart_buildings.smart_control.environment import hybrid_action_environment

    env = hybrid_action_environment.HybridActionEnvironment()
    env.metrics_path = None
    meta = ev.create_building_meta_data(env, d)
    spec = env.action_spec()
    layout = build_layout(meta)
    captured = {}
    orig = env.reward_function.compute_reward

    def wrapped(info):
        r = orig(info)
        captured["r"], captured["i"] = r, info
        return r

    env.reward_function.compute_reward = wrapped
    cls = load_class(policy_spec)
    policy = cls(meta, spec, params) if params else cls(meta, spec)
    acc = {p: {k: 0.0 for k in PARTS + EQUIP + ("steps",)} for p in PERIODS}
    obs, _ = env.reset()
    occ_min, n = np.inf, 0
    h = 300.0 / 3600.0
    for _ in range(steps):
        s = decode(obs, layout)
        occ_min = min(occ_min, s.occupancy)
        period = "weekend" if s.dow >= 5 else ("occupied" if s.occupancy > occ_min + 1e-4 else "empty_workday")
        obs, _, term, trunc, _ = env.step(ev.validate_action(policy.get_action(obs), spec))
        r, info = captured["r"], captured["i"]
        a = acc[period]
        a["steps"] += 1
        a["comfort"] += 0.6 * r.normalized_productivity_regret
        a["energy"] -= 0.2 * r.normalized_energy_cost
        a["carbon"] -= 0.2 * r.normalized_carbon_emission
        a["fan_kWh"] += h * sum(v.blower_electrical_energy_rate for v in info.air_handler_reward_infos.values()) / 1e3
        a["chiller_kWh"] += h * sum(abs(v.air_conditioning_electrical_energy_rate)
                                    for v in info.air_handler_reward_infos.values()) / 1e3
        a["pump_kWh"] += h * sum(v.pump_electrical_energy_rate for v in info.boiler_reward_infos.values()) / 1e3
        a["gas_kWh"] += h * sum(v.natural_gas_heating_energy_rate for v in info.boiler_reward_infos.values()) / 1e3
        n += 1
        if term or trunc:
            break
    for p in PERIODS:  # reward parts as mean per step over the whole episode
        for k in PARTS:
            acc[p][k] /= n
    return {"scenario": f"{d.parent.name}/{d.name}", "steps": n, "periods": acc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--params")
    ap.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--include", nargs="*")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text()) if a.params else None
    scen = discover(Path(a.scenarios), a.include)
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(a.workers) as pool:
        for f in as_completed([pool.submit(run, str(s), a.policy, params, a.steps) for s in scen]):
            res = f.result()
            out.append(res)
            print(f"\n{res['scenario']} ({res['steps']} steps)")
            print(f"  {'period':14s} {'share':>6s} {'comfort':>8s} {'energy':>8s} {'carbon':>8s} "
                  f"{'fan':>8s} {'chiller':>8s} {'pump':>7s} {'gas':>8s}  (kWh)")
            for p, v in res["periods"].items():
                print(f"  {p:14s} {v['steps'] / res['steps']:6.1%} {-v['comfort']:8.5f} {-v['energy']:8.5f} "
                      f"{-v['carbon']:8.5f} {v['fan_kWh']:8.0f} {v['chiller_kWh']:8.0f} {v['pump_kWh']:7.0f} "
                      f"{v['gas_kWh']:8.0f}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{a.name}.json").write_text(json.dumps({"policy": a.policy, "params": params, "results": out}, indent=1))
    print(f"\nwrote {RUNS / (a.name + '.json')} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
