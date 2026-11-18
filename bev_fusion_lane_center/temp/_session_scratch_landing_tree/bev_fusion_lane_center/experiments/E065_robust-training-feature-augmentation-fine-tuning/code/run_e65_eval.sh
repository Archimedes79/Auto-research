#!/usr/bin/env bash
# E65: ein Kopf (Argument 1) auf sauberen (stored) und allen gestoerten TEST-Merkmalen (in der Reihenfolge der Fertigstellung)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
H=$1
echo "== $(date +%H:%M:%S) eval $H stored"
./run_gcar.sh eval_pert.py --pert stored --head $H --workers 4 2>&1 | grep -E "^\[|Traceback|Error"
./run_gcar.sh eval_pert.py --pert stored --stride 2 --head $H --workers 4 2>&1 | grep -E "^\[|Traceback|Error"
PENDING="dark cam_front_off cam_front3_off blur extr3 fog radar_off"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ -f $HOME/exp_gcar_b123/pert/$P/DONE ]; then
      echo "== $(date +%H:%M:%S) eval $H $P"
      ./run_gcar.sh eval_pert.py --pert $P --head $H --workers 4 2>&1 | grep -E "^\[|Traceback|Error"
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) eval $H done"
