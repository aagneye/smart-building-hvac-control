# Results log

Scores are the leaderboard metric (mean |per-step reward|, lower is better),
computed with `src/sb/eval/harness.py` on the current starter kit.

## Full 30 training scenarios, 30 days

| Date | Policy | Params | Score | Notes |
|---|---|---|---|---|
| 2026-10-30 | pid_water (kit baseline) | — | 0.0550 | reference |
| 2026-10-30 | ControllerV1 | defaults (Kp 0.07, Ki 0.0035) | 0.0194 | comfort ≈ 0.0005; energy dominates, Fairbanks worst |
| 2026-10-30 | ControllerV2 | `params/v2_tuned_g22.json` (CMA-ES run tune_v2_a) | 0.00788 | comfort ≈ 0.0003; slowest step 1.26 ms |

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
| 2026-10-30 | ControllerV2, regime-based supply air | 0.0173 | beats v1 on all 6; Fairbanks unchanged at 0.067 |

Per station, tuned v2 vs v1 (full 30 days): atlanta 0.0050 vs 0.0118, denver
0.0044 vs 0.0142, fairbanks 0.0163 vs 0.0512, phoenix 0.0076 vs 0.0122,
singapore 0.0061 vs 0.0079.

## Gateway parity

The built notebook run through the real Kaggle gateway (competition data,
`scripts/test_notebook_gateway.py`) gave −0.000685 on the first day of
building_train_1/atlanta_2; the harness gives −0.00069 on the same day.

## Kaggle submissions

| Date | Notebook version | Policy | Params | Public LB | Notes |
|---|---|---|---|---|---|
| 2026-10-30 | aagneye/sb-hvac-controller-v2 v1 | ControllerV2 | `params/v2_tuned_g22.json` | 0.01129 | submission 57814383, vs 0.00788 on the training 30; commit run scored −0.003703 on the example scenario (harness −0.00369) |

## Wide holdout: unseen buildings and unseen climates (first week)

Set from `scripts/make_wide_set.py`: 4 new training-sized buildings x 3 held-out
synthetic climates (hot tropic, high altitude, mild marine) + 2 large buildings.

| Date | Policy | Params | Holdout score |
|---|---|---|---|
| 2026-10-31 | ControllerV2 | defaults | 0.00773 |
| 2026-10-31 | ControllerV2 | `params/v2_tuned_g22.json` (submitted) | 0.00642 |

Worst holdout scenario: large building_0502 in the hot tropic climate (0.0213,
comfort part 0.0039). Raising the per-zone cooling cap `q_cool_max_W` from
9.7 kW to 20 kW cut it to 0.0183 with comfort 0.0002: a fixed watt cap tuned
on small rooms under-serves big rooms in hot weather.

Tuning run `tune_v2_wide` started 2026-10-31 from the submitted set on 32 wide
training scenarios (12 new buildings x 2 climates incl. synthetic desert and
subarctic, + 4 training buildings x 2 unseen strips); start score 0.01293.
