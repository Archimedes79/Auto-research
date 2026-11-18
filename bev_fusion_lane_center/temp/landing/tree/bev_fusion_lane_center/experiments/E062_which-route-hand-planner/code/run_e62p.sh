#!/usr/bin/env bash
# E62p (Voraussetzung fuer E64): planer-nahe Auswertung (sigma 1 m, Registrierung) auf den Graphen jeder gestoerten Bedingung -> out/loop2/e62p_<bed>/persample_test.pkl; Reihenfolge der Fertigstellung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e61[.]sh|run_e62[.]sh" > /dev/null; do sleep 30; done
if [ ! -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e57_stored/graphs_test.pkl ]; then
  echo "== $(date +%H:%M:%S) stored (Graphen)"
  ./run_gcar.sh eval_pert.py --pert stored --workers 3 2>&1 | grep -E "^\[|Traceback|Error"
fi
PENDING="stored dark cam_front_off cam_front3_off blur extr3 fog radar_off"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e57_$P/graphs_test.pkl ]; then
      echo "== $(date +%H:%M:%S) plan_eval $P"
      ./run_gcar.sh plan_eval.py --sets test --graphs e57_$P/graphs --sigma 1.0 --reg 1 --workers 4 --out e62p_$P 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-300
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) done"
