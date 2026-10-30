# Smart Buildings HVAC control

Our entry for the NeurIPS 2026 Smart Buildings Kaggle competition: one HVAC
control policy that must work on buildings and climates it has never seen.

Approach: a physically consistent, layered controller whose constants are tuned
with CMA-ES on many generated buildings, with online system identification for
pre-conditioning added later. See [docs/data-and-physics.md](docs/data-and-physics.md)
for the data, preprocessing and plant physics.

## Layout

```
src/sb/obs.py           decode the observation vector into physical units
src/sb/params.py        every tunable constant, with search bounds
src/sb/policy.py        the controller (ControllerV1, exported as MyPolicy)
src/sb/eval/harness.py  parallel full-episode evaluation, leaderboard metric
scripts/sync.sh         push local code to the VM
scripts/trace.py        step-level trace of one policy on one scenario
scripts/check_obs.py    verify the observation decoder
params/                 parameter sets (JSON overrides of params.py defaults)
docs/                   notes
```

## Running

The simulator is CPU-bound and runs on the Azure VM in `/data/smartbuild`,
with the official starter kit at `/data/smartbuild/starter-kit`.

```bash
bash scripts/sync.sh
```

```bash
ssh azureuser@20.51.160.183
cd /data/smartbuild && . .venv/bin/activate
python -m sb.eval.harness --policy sb.policy:ControllerV1 --params params/lowgain.json --workers 30 --name v1
```

Add `--include building_train_8 --steps 2016` for a quick one-week check on
one building. Results land in `runs/<name>.json`.
