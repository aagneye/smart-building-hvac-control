# Results log

Scores are the leaderboard metric (mean |per-step reward|, lower is better),
computed with `src/sb/eval/harness.py` on the current starter kit.

## Full 30 training scenarios, 30 days

| Date | Policy | Params | Score | Notes |
|---|---|---|---|---|
| 2026-10-30 | pid_water (kit baseline) | — | 0.0550 | reference |
| 2026-10-30 | ControllerV1 | defaults (Kp 0.07, Ki 0.0035) | 0.0194 | comfort ≈ 0.0005; energy dominates, Fairbanks worst |

Per station, ControllerV1 vs pid_water: atlanta 0.0118 vs 0.0306, denver
0.0142 vs 0.0379, fairbanks 0.0512 vs 0.0791, phoenix 0.0122 vs 0.0800,
singapore 0.0079 vs 0.0471.

## Quick check: first week of 6 scenarios

Scenarios: train_1/fairbanks_1, train_2/phoenix_6, train_4/singapore_3,
train_5/atlanta_6, train_7/denver_5, train_8/phoenix_4 (2016 steps).

| Date | Policy | Score | Notes |
|---|---|---|---|
| 2026-10-30 | pid_water | 0.1038 | |
| 2026-10-30 | ControllerV1, high gains (Kp 0.35) | — | oscillated between full heat and cool |
| 2026-10-30 | ControllerV1, defaults | 0.0319 | |
| 2026-10-30 | ControllerV2 | 0.0198 | Fairbanks 0.145 → 0.067; Phoenix worse than v1 |
