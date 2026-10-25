#!/usr/bin/env bash
# E57: ausgewertet wird jede fertig extrahierte Stoerung (wartet auf DONE)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for P in "$@"; do
  until [ -f $HOME/exp_gcar_b123/pert/$P/DONE ]; do sleep 20; done
  echo "== $(date +%H:%M:%S) eval $P"
  ./run_gcar.sh eval_pert.py --pert $P --workers 4 2>&1 | grep -E "^\[|Traceback|Error|fertig"
done
echo "== $(date +%H:%M:%S) eval done"
