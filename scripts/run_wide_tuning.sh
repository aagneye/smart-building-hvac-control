#!/usr/bin/env bash
# Launch the wide tuning run on the VM, detached so it survives disconnects.
#
#   ssh azureuser@20.51.160.183 'bash /data/smartbuild/scripts/run_wide_tuning.sh'
#   ssh azureuser@20.51.160.183 'bash /data/smartbuild/scripts/run_wide_tuning.sh --resume'   # after a reboot
#   ssh azureuser@20.51.160.183 'bash /data/smartbuild/scripts/run_wide_tuning.sh --tune'     # tuner only
#
# First scores the submitted parameters and the defaults on the holdout set
# (unseen buildings and unseen climates), then tunes from the submitted set.
# Logs: logs/wide_holdout_*.log, logs/tune_v2_wide.log; results: runs/tune_v2_wide/.
set -euo pipefail
cd /data/smartbuild
. .venv/bin/activate
S=/data/smartbuild/starter-kit/smart_buildings/wide_scenarios
TRAIN="building_0700 building_0704 building_0705 building_0706 building_0709 building_0711 building_0713
       building_0714 building_0717 building_0719 building_0721 building_0723
       building_train_1 building_train_4 building_train_7 building_train_8"
HOLDOUT="building_0724 building_0725 building_0727 building_0729 building_0502 building_0511"
ONLY="margin_lo_K margin_hi_K lag_comp kp ki i_leak sat_cool_K sat_econ_K ahu_on_thr ahu_off_thr
      ahu_min_on_steps econ_margin_K tw_cold_K tw_warm_K oat_cold_K oat_warm_K dp_min_frac
      rate_heat_K_per_h rate_cool_K_per_h prestart_buffer_h prestart_max_h unocc_lo_K unocc_hi_K
      holiday_check_h q_heat_max_W q_cool_max_W dis_margin_K water_margin f_floor"
TUNE="python -m sb.eval.tune_cmaes --name tune_v2_wide --policy sb.policy_v2:ControllerV2 --scenarios $S
      --train $TRAIN --holdout $HOLDOUT --steps 2016 --popsize 12 --sigma 0.2 --workers 58
      --generations 30 --holdout-every 5 --start params/v2_tuned_g22.json --only $ONLY"
# One line, so it survives being embedded in `bash -c "..."` below.
TUNE=$(echo $TUNE)
mkdir -p logs runs

if [ "${1:-}" = "--resume" ]; then
  nohup $TUNE --resume >> logs/tune_v2_wide.log 2>&1 < /dev/null &
  echo "resumed tune_v2_wide"
  exit 0
fi
if [ "${1:-}" = "--tune" ]; then  # tuner only, holdout baselines already done
  nohup $TUNE > logs/tune_v2_wide.log 2>&1 < /dev/null &
  echo "started tune_v2_wide"
  exit 0
fi

BASE="python -m sb.eval.harness --policy sb.policy_v2:ControllerV2 --scenarios $S --include $HOLDOUT --steps 2016 --workers 16"
nohup bash -c "
  $BASE --params params/v2_tuned_g22.json --name wide_holdout_g22 > logs/wide_holdout_g22.log 2>&1 &
  $BASE --name wide_holdout_defaults > logs/wide_holdout_defaults.log 2>&1 &
  wait
  $TUNE > logs/tune_v2_wide.log 2>&1
" > logs/wide_launch.log 2>&1 < /dev/null &
echo "launched holdout baselines, then tune_v2_wide"
