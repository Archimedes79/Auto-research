#!/usr/bin/env bash
# E57: zweite Runde (Reihenfolge nach Wichtigkeit); wartet auf das Ende der laufenden Extraktion des ersten Strangs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
WAIT=$1; shift
until [ -f $HOME/exp_gcar_b123/pert/$WAIT/DONE ]; do sleep 15; done
for P in "$@"; do
  echo "== $(date +%H:%M:%S) $P"
  ./run_gcar.sh gcar_pert.py --pert $P --workers 3 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
done
echo "== $(date +%H:%M:%S) done"
