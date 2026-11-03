# Plan: closing the gap to the top 10

Written 2026-11-03. Data: `docs/results.md`, runs `eb_v2_g22`, `fb1_week_v2/v3`,
`wide_holdout_*` on the VM.

## Where we stand

| | Public LB score | Rank (of 526) |
|---|---|---|
| 1st | 0.00470 | 1 |
| 10th (target) | 0.00568 | 10 |
| 50th | 0.00720 | 50 |
| **Us: v2, tuned set g22** | **0.01129** | **210** |

The same controller scores 0.00788 on the 30 training scenarios, so the hidden
test costs us about 43% more. Reaching the top 10 means roughly halving our
public score. Top-10 teams have a median of 22 submissions; we have made 1.

## What the data says

1. **Energy, not comfort, is our loss on known climates.** Over 8 training
   scenarios (30 days, all 5 climates): in hot and mild climates 84% of the loss
   is energy used while people are in, comfort is 2%. In Fairbanks, 58% is
   daytime energy, about a third is heating at night and on weekends, 9% is
   comfort.
2. **In hot climates the chiller is almost all of it.** Chiller electricity is
   about 15x fan electricity. With a fixed COP, the honest levers are smaller:
   less outside air per unit of cooling (already at the coldest supply air),
   warmer zones within the band, free cooling at night, and shifting cooling to
   cheaper and cleaner hours.
3. **In the cold we are already near the frontier.** First week of
   train_1/fairbanks_1: ours −91.1, a forum team's planner −85.4, their LP lower
   bound on the loss −79.1. We are 7% behind a published planner and 15% above
   the bound in the hardest scenario.
4. **Unseen buildings and climates expose a capacity limit.** Our zone heating
   and cooling caps are fixed watts tuned on small rooms. A big-room building in
   a hot climate lost 0.0039 to comfort; a bigger cap removed most of it. v3
   (adaptive per-zone capacity) fixes the heat case (holdout 0.00642 → 0.00624)
   but, at a 60% growth trigger, it badly hurts heating in the cold (Fairbanks
   week −91.1 → −153.3). It must not be submitted until heating is safeguarded.
5. **Retuning on more of the same does not generalize.** The wide CMA-ES run
   improved its training set 7% but made the holdout 11% worse. Gains must be
   checked on unseen buildings and climates and, ideally, on the leaderboard.
6. **Part of the public top 10 may be exploit-driven.** Forum threads report
   that sawtoothing the hot-water setpoint fakes about 26% energy and up to half
   the carbon in cold climates, and that the area-weighted return air creates
   artificial heat. The organizers plan to re-run the top submissions on a
   corrected simulator with a different occupancy seed. These scores are likely
   to fall then; a physically consistent controller keeps its score.

## Why we are not in the top 10 (best estimate)

| Source | Evidence | Rough share of our gap |
|---|---|---|
| Test-only failures (big rooms, extreme heat or cold, unseen buildings) | train→test +43%; holdout comfort concentrated in big rooms in heat | large, unknown; measure with LB probes |
| Control quality on known climates | cold: 15% above an LP bound; warm: limited honest levers | 10–20% of our score |
| Rivals using simulator quirks | forum reports, organizer re-run plans | possibly a large part of the top-10 margin |

Honest improvements alone probably take us to roughly 0.008–0.009 on the
current public board (top 50–80). The top 10 on the public board likely also
needs either a step change in control or the quirks we choose not to use. The
final ranking is after the re-run, which favors a clean controller.

## The plan

Every candidate passes the same gates before it is submitted:
full 30-day run on the official 30, the wide holdout (unseen buildings and
climates), and the Fairbanks week against the −85.4 / −79.1 references. No
candidate may be worse than v2-g22 on any of the three by more than 2%.

### Phase 1 — this week: fix v3 and start measuring the hidden test

1. **v3 cooling-only.** Apply adaptive capacity to cooling only, and grow the
   heating cap only on true saturation with a lower limit. Pass the gates;
   submit. Expected: fewer comfort failures in hot unseen climates.
2. **Leaderboard probes, 2–3 submissions a day.** Each changes one thing from
   the submitted v2-g22 so its effect on the hidden test is visible:
   wider comfort margins (+0.3 K), v2 defaults (overfitting check), v3
   cooling-only, earlier pre-start. The deltas tell us whether the test loses
   to comfort, to tuning, or to capacity.
3. **Re-measure every submission's commit run** and record it in
   `docs/results.md` with the public score, to calibrate which local set best
   predicts the leaderboard.

### Phase 2 — weeks 2–3: model-based scheduling

4. **Online zone model.** Fit each zone's response (outside air, supply heat,
   neighbors) by recursive least squares from the first days of an episode.
5. **Optimal start and setback.** Start each AHU's morning recovery at the
   latest time that reaches the band, and choose the night setback depth that
   minimizes heat lost plus recovery cost. Targets: the night/weekend third of
   cold-climate loss and the 9% comfort loss in Fairbanks.

### Phase 3 — weeks 3–5: cooling-climate energy

6. **Free and cheap cooling.** Night flushing with the economizer when outside
   air is cooler than the slab, pre-cooling before 06:00 at the cheaper
   night tariff, riding the upper band edge late in the day.
7. **Tank standing losses.** Drop the hot-water setpoint when no zone has
   needed heat for a day (two thirds of warm-climate gas is burned with nobody
   in).
8. **Tune with a guard.** CMA-ES on the wide set plus unused official strips,
   full months, keeping the distribution mean and stopping when the holdout
   stops improving.

### Phase 4 — weeks 5–8: planner

9. **Hourly plan per AHU over 24 hours** using the identified model and a
   weather forecast from the episode's own history (an LP or small MPC), with
   the rule-based controller as the inner loop. This is what the forum's
   planner team does in the cold; we would extend it to cooling.

### Phase 5 — weeks 9–10: hardening and final picks

10. Test on PR #2's corrected boiler, on other occupancy seeds, and on the
    largest generated buildings for the 9-hour limit. Pick the two final
    submissions by corrected-simulator score.

## Guardrails

- No exploitation of simulator quirks; the re-run would erase those gains.
- No building- or weather-specific constants.
- Long runs at low CPU priority on the shared VM, resumable, results committed.
