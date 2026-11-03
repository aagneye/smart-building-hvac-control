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
| 2026-11-03 | aagneye/sb-hvac-probe-margins v1 | ControllerV2 | `params/probe_v2_margins03.json` (g22, margins +0.3 K) | pending | submission 57953403; probe: does the hidden test punish comfort? example scenario −0.0040 |
| 2026-11-03 | aagneye/sb-hvac-probe-defaults v1 | ControllerV2 | defaults | pending | submission 57953404; probe: does tuning overfit? example scenario −0.0057 |
| 2026-11-03 | aagneye/sb-hvac-v3 v1 | ControllerV3 | `params/v3_g22_cool.json` | pending | submission 57954055; training-30 0.007879, wide holdout 0.006185, Fairbanks week −91.1 (same as v2); example scenario −0.0037 |

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

## 2026-11-02/03 measurements

- `tune_v2_wide` finished 30 generations: training 0.01293 → 0.01205, but its
  best scored 0.00715 on the wide holdout vs 0.00642 for the submitted set
  (overfit). Its distribution mean scored 0.00645.
- ControllerV3 (adaptive per-zone capacity) with the submitted set, wide
  holdout: 0.00624 (v2 0.00642). On the three hardest holdout scenarios the
  growth trigger matters: 0.98 → 0.01532, 0.8 → 0.01523, 0.6 → 0.01498
  (v2 0.01622).
- First week of train_1/fairbanks_1, total reward: v2-g22 −91.1, v3 at a 0.6
  trigger −153.3 (heating regression), forum planner −85.4, forum LP bound −79.1.
- Loss breakdown, v2-g22, 8 official scenarios x 30 days (`scripts/energy_breakdown.py`):
  warm climates 84% of loss is energy while occupied, 2% comfort, chiller about
  15x fan electricity; cold climates 58% daytime energy, about a third night and
  weekend heating, 9% comfort.
- Public leaderboard 2026-11-03: 10th 0.00568, 50th 0.00720, ours 0.01129 (210th of 526).

## Phase 1 calibration (local, official 30, 30 days)

| Variant | Local score | vs v2-g22 | Public LB |
|---|---|---|---|
| v2-g22 (submitted 2026-10-30) | 0.007877 | — | 0.01129 |
| Probe: margins +0.3 K | 0.008152 | +3.5% | pending |
| Probe: v2 defaults | 0.011241 | +43% | pending |
| v3 cooling-first | 0.007879 | +0.0% | pending |

Reading the probes when scores arrive: a margins LB below 0.01129 means the
hidden test loses comfort; a defaults LB much less than 43% worse means the
tuning overfits; a v3 LB below 0.01129 means big rooms in heat matter there.
