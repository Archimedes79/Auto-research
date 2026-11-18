#!/usr/bin/env bash
# E57: wertet jede fertig extrahierte Stoerung aus (in der Reihenfolge der Fertigstellung, wartet auf DONE); Argumente: Stoerungen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
PENDING="$@"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ -f $HOME/exp_gcar_b123/pert/$P/DONE ] && [ ! -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e57_$P/metrics.json ]; then
      echo "== $(date +%H:%M:%S) eval $P"
      ./run_gcar.sh eval_pert.py --pert $P --workers 4 2>&1 | grep -E "^\[|Traceback|Error|fertig"
    elif [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e57_$P/metrics.json ]; then
      :
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) eval done"
