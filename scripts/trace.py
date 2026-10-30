"""Step-level trace of one policy on one scenario, for debugging.

    python scripts/trace.py building_train_8/denver_6 576 [module:Class] [params.json]

Writes runs/trace_<building>_<weather>.csv with one row per step and prints an
hourly summary for the first days.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path("/data/smartbuild/starter-kit")
os.chdir(KIT)
from smart_buildings import evaluate as ev  # noqa: E402

from sb.eval.harness import load_class  # noqa: E402
from sb.obs import build_layout, decode  # noqa: E402

scen = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 576
spec = sys.argv[3] if len(sys.argv) > 3 else "sb.policy:ControllerV1"
params = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None

d = KIT / "smart_buildings/training_scenarios" / scen
env = ev.load_hybrid_environment(d / "config.gin")
meta = ev.create_building_meta_data(env, d)
L = build_layout(meta)
captured = {}
orig = env.reward_function.compute_reward


def wrapped(info):
    r = orig(info)
    captured["r"], captured["i"] = r, info
    return r


env.reward_function.compute_reward = wrapped
cls = load_class(spec)
pol = cls(meta, env.action_spec(), params) if params else cls(meta, env.action_spec())
obs, _ = env.reset()
rows = []
for k in range(steps):
    s = decode(obs, L)
    a = pol.get_action(obs)
    c, disc = a["continuous_action"].astype(float), a["discrete_action"]
    obs, rew, *_ = env.step(ev.validate_action(a, env.action_spec()))
    r, info = captured["r"], captured["i"]
    elec_fan = sum(v.blower_electrical_energy_rate for v in info.air_handler_reward_infos.values())
    elec_cool = sum(v.air_conditioning_electrical_energy_rate for v in info.air_handler_reward_infos.values())
    pump = sum(v.pump_electrical_energy_rate for v in info.boiler_reward_infos.values())
    gas = sum(v.natural_gas_heating_energy_rate for v in info.boiler_reward_infos.values())
    rows.append(dict(
        step=k, hour=round(s.hour, 2), dow=s.dow, occ=s.occupancy, oat=s.oat - 273.15,
        tz_min=s.zone_t.min() - 273.15, tz_mean=s.zone_t.mean() - 273.15, tz_max=s.zone_t.max() - 273.15,
        lo=s.zone_lo.mean() - 273.15, hi=s.zone_hi.mean() - 273.15, hws=s.hws_t - 273.15,
        ahu_on=int(disc[:L.K].sum()), pump=int(disc[L.K]),
        sat=(0.5 * (c[L.a_ahu_temp] + 1) * 20 + 285 - 273.15).mean(),
        sp=(0.5 * (c[L.a_ahu_pressure] + 1) * 2000).mean(),
        recirc=(0.5 * (c[L.a_ahu_recirc] + 1)).mean(),
        dp=0.5 * (c[L.a_pump_dp] + 1) * 20, tw=0.5 * (c[L.a_water_temp] + 1) * 40 + 310 - 273.15,
        damper=(0.5 * (c[L.a_damper] + 1)).mean(), valve=(0.5 * (c[L.a_reheat] + 1)).mean(),
        fan_kW=elec_fan / 1e3, cool_kW=elec_cool / 1e3, pump_kW=pump / 1e3, gas_kW=gas / 1e3,
        reward=float(rew), comfort=0.6 * r.normalized_productivity_regret,
        energy=-0.2 * r.normalized_energy_cost, carbon=-0.2 * r.normalized_carbon_emission,
    ))
df = pd.DataFrame(rows)
out = Path("/data/smartbuild/runs") / f"trace_{scen.replace('/', '_')}_{spec.split(':')[-1]}.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
df["day"] = df.step // 288
hourly = df.groupby([df.day, df.hour.astype(int)]).mean(numeric_only=True)
print(hourly[["occ", "oat", "tz_min", "tz_mean", "tz_max", "lo", "hi", "ahu_on", "pump", "sat", "sp", "recirc",
              "dp", "tw", "hws", "damper", "valve", "fan_kW", "cool_kW", "pump_kW", "gas_kW", "reward", "comfort"]].round(2).to_string())
print("totals:", df[["reward", "comfort", "energy", "carbon"]].mean().round(5).to_dict())
print("wrote", out)
