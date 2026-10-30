#!/usr/bin/env bash
# Regenerate our extra scenario sets inside the starter kit on the VM.
#
#   tune_scenarios/  training buildings x weather strips they never see in the
#                    official set (6 stations incl. minneapolis). Fast to
#                    simulate; used for CMA-ES. train_1/4/7/8 tune, train_2/5 hold out.
#   gen_scenarios/   11 new generator buildings (seeds 500-511, 503 is rejected
#                    by the generator) x 6 stations. 2-9x larger grids, so slow;
#                    used for transfer checks.
#
# Run on the VM: bash scripts/make_scenarios.sh
set -euo pipefail
cd /data/smartbuild/starter-kit
. ../.venv/bin/activate
export OMP_NUM_THREADS=1
stations() { echo "fairbanks_$1 denver_$1 atlanta_$1 phoenix_$1 singapore_$1 minneapolis_$1"; }

for pair in train_1:3 train_4:6 train_7:4 train_8:2 train_2:5 train_5:7; do
  b=building_${pair%%:*}; n=${pair##*:}
  python -m smart_buildings.run_scenario_generator --buildings "$b" --weathers $(stations "$n") \
    --output-dir smart_buildings/tune_scenarios > /dev/null &
done
wait

python -m smart_buildings.run_floor_generator --start-seed 500 --count 12 \
  --output-dir smart_buildings/buildings_gen --no-pdf > /dev/null
for i in $(seq 0 11); do
  b=$(printf "building_%04d" $((500 + i)))
  [ "$b" = building_0503 ] && continue  # room below the 20 m2 minimum
  python -m smart_buildings.run_scenario_generator --buildings "smart_buildings/buildings_gen/$b.yaml" \
    --weathers $(stations $(( (i % 8) + 1 ))) --output-dir smart_buildings/gen_scenarios > /dev/null &
done
wait
echo "tune: $(find smart_buildings/tune_scenarios -name config.gin | wc -l)  gen: $(find smart_buildings/gen_scenarios -name config.gin | wc -l)"
