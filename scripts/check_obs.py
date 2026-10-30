"""Roll pid_water for N steps on one scenario and check the decoder.

Prints the physical range of every decoded quantity and asserts that hour and
weekday decode to the simulator's own clock. Run from /data/smartbuild:

    python scripts/check_obs.py building_train_8/denver_6 2016
"""

import os
import sys
from pathlib import Path

import numpy as np

KIT = Path("/data/smartbuild/starter-kit")
os.chdir(KIT)
from smart_buildings import evaluate as ev  # noqa: E402
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy  # noqa: E402

from sb.obs import build_layout, decode  # noqa: E402

scen = sys.argv[1] if len(sys.argv) > 1 else "building_train_8/denver_6"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
d = KIT / "smart_buildings/training_scenarios" / scen
env = ev.load_hybrid_environment(d / "config.gin")
meta = ev.create_building_meta_data(env, d)
L = build_layout(meta)
policy = PIDWaterResetPolicy(meta, env.action_spec())
obs, _ = env.reset()
rec = {k: [] for k in ["zone_t", "zone_lo", "zone_hi", "damper", "reheat", "oat", "sat", "hws_t", "occupancy"]}
bad_clock = 0
for t in range(steps):
    s = decode(obs, L)
    ts = env.current_simulation_timestamp
    ts = ts.tz_convert(meta["time_zone"]) if ts.tzinfo is not None else ts
    true_hour = ts.hour + ts.minute / 60.0
    if abs(((s.hour - true_hour + 12) % 24) - 12) > 0.05 or s.dow != ts.dayofweek:
        bad_clock += 1
        if bad_clock <= 3:
            print("clock mismatch", t, s.hour, s.dow, ts)
    for k in rec:
        rec[k].append(np.ravel(getattr(s, k)))
    obs, *_ = env.step(ev.validate_action(policy.get_action(obs), env.action_spec()))
print(f"V={L.V} K={L.K} steps={steps} clock_mismatches={bad_clock}")
for k, v in rec.items():
    a = np.concatenate(v)
    print(f"{k:10s} min {a.min():9.3f}  p50 {np.median(a):9.3f}  max {a.max():9.3f}")
