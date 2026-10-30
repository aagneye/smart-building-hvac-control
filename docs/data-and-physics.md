# Data, preprocessing and plant physics

Reference notes for the Smart Buildings competition. Sources: the official
starter kit (`smart_buildings/`), the Kaggle pages, and our own checks on the VM.

## There is no dataset

The competition scores a **control policy**, not predictions. Kaggle's data tab
holds only the evaluation gateway. Everything else comes from the starter kit:

| Item | What it is |
|---|---|
| Simulator | `smart_buildings/smart_control/`, a fork of Google's sbsim |
| Buildings | 6 training buildings (`buildings/building_train_{1,2,4,5,7,8}.yaml`), 28 to 85 zones, 3 to 9 air handlers |
| Weather | 2023 hourly station data for fairbanks, denver, atlanta, phoenix, singapore (+ minneapolis) |
| Scenarios | 30 = each building × one 30-day strip per station, `training_scenarios/<building>/<strip>/config.gin` |
| Generators | `run_floor_generator.py` and `run_scenario_generator.py` make unlimited new buildings and scenarios |

The hidden test set is 6 unseen buildings from the same generator × 5 unseen
weather stations. Our "training data" is therefore whatever we simulate.

## Episode and score

- 8,640 steps of 5 minutes = 30 days, every step scored.
- Reward per step: `0.6·comfort − 0.2·energy_cost − 0.2·carbon`, always ≤ 0.
- Score: mean over scenarios of |mean per-step reward|. Lower is better.
- Comfort per zone: deviation outside the band costs `dev/5` up to 5 K, then
  `1 + ((dev−5)/5)²`, unbounded. Averaged over **occupants**, so it is exactly
  zero when nobody is in the building.

## Observation (flat float vector, length 6V + 15K + 58)

V = zones, K = air handlers. Fields are found by name from
`building_meta_data["observation_field_names"]`, never by position.

| Field group | Scaling as delivered | Decoded to |
|---|---|---|
| `VAV_n_zone_air_temperature_sensor` | normalized, `(T − 295)/5` | kelvin |
| `VAV_n_supply_air_damper…`, `…reheat_valve…` | normalized, `(u − 0.5)/0.5` | 0..1 as realized |
| `VAV_n_zone_air_{heating,cooling}_temperature_setpoint_sensor` | raw kelvin | comfort band edges |
| `ahs_<ahu>_*` (supply temp, pressure, flows, outside air temp) | raw units | as is |
| `hws_supply_water_temperature_sensor` | raw kelvin | as is |
| `hod_{sin,cos}_000..011`, `dow_{sin,cos}_000..011` | 12 phase-shifted pairs | hour 0..24, weekday 0=Mon |
| `num_occupants` | raw head count (building total) | as is |
| `comfort_mode_now`, `comfort_mode_soon` | 0/1 | flags |

Gotchas:

- Zone temperatures lag one step behind the simulator.
- The AHU supply-air temperature sensor reports the **setpoint**, not the real
  supply, when the mixed air is already colder.
- There is no weather label, building id, humidity, or per-zone occupancy.
- The normalization metadata lists stats for fields that arrive raw, so do not
  apply it blindly.

`src/sb/obs.py` implements the decoding; `scripts/check_obs.py` verifies it
against the simulator clock (0 mismatches over a 7-day rollout).

## Action

`{"continuous_action": float32[3K+2+2V] in [−1,1], "discrete_action": int32[K+1]}`

| Slot | Native range | Notes |
|---|---|---|
| AHU run (discrete) | 0/1 | 0 = no air at all, zones free-float |
| AHU supply-air temperature | 285–305 K | can only cool: supply = min(mixed air, setpoint) |
| AHU static pressure | 0–2000 Pa | floored at 200 Pa while running |
| AHU recirculation | 0–1 | capped at 0.7, so at least 30% outside air |
| Pump run (discrete) | 0/1 | 0 = no hot water, no reheat anywhere |
| Pump differential pressure | 0–20 bar | last entry of `pressure_indices` |
| Supply water temperature | 310–350 K | `water_indices` |
| VAV damper / reheat valve | 0–1 | damper floored at 0.1 |

Zero in normalized units is mid-range, not off. Indices come from
`temp_indices`, `pressure_indices`, `recirculation_indices`, `water_indices`
and `zone_action_indices` in the metadata.

## Plant physics that shape the controller

- **All heat comes from VAV reheat.** In cold weather, every bit of air moved
  carries at least 30% outside air and becomes heating load.
- **Zone air reacts in about 2.5 minutes at full airflow**, faster than the
  5-minute step. With the one-step sensor lag, high-gain zone control
  oscillates between full heating and full cooling. Use low gains.
- **Airflow** per zone = `damper · 2.0 m³/s · sqrt(static pressure / 2000)`.
  Fan power scales with flow × pressure, so open dampers and lower pressure.
- **Pump power** scales with water flow × dp, so open valves and lower dp.
- **Electricity costs about 2.3× more reward per kW than gas.**
- **Comfort only counts when occupants are present:** workdays from 06:00
  (arrivals until noon) to 19:00. Nights, weekends and holidays are free, as
  long as zones are back in band by 06:00.
- **Boiler bug:** gas is billed at the setpoint while coils draw at the tank
  temperature, so sawtoothing the setpoint fakes savings. The organizers plan
  to re-run the top submissions on a corrected simulator, so we keep the water
  setpoint smooth and never rely on it.

## Baseline reference (current kit, our harness)

| Policy | Score on 30 training scenarios |
|---|---|
| pid_water | 0.0550 |

The kit's `baseline_returns.json` (0.0541) predates the tracker fix in PR #1;
the current official evaluator reproduces our harness numbers exactly.
