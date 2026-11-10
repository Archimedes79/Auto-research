#!/bin/bash
# scratch: the safety check config by config, old then new (resumable: finished configs are skipped); log in safety_all.log
HERE="$(cd "$(dirname "$0")" && pwd)"
for label in t11_vel_hurdle_acc2 t10_dual_s0 t10_dual_s1 t8_ctl_s0 t8_ctl_s1 t16_full.json t20_base.json t16_full_s0 t20_base_s0; do
  bash "$HERE/run_safety.sh" old "$label"
  bash "$HERE/run_safety.sh" new "$label"
done
echo "ALL CONFIGS DONE"
