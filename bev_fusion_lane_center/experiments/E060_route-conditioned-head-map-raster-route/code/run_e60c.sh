#!/usr/bin/env bash
# E60: nach den Kontrollen - (1) Anpassung an das Training (TRAIN-eval), (2) der Kopf mit Karte auf den gestoerten TEST-Merkmalen (E57) in der Reihenfolge der Fertigstellung: rettet die Karte die Wahrnehmung?
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e61[.]sh|run_e62[.]sh" > /dev/null; do sleep 30; done
echo "== $(date +%H:%M:%S) train-eval e60_main"
./run_gcar.sh route_head.py --name e60_main --eval_only 1 --eval train --conds clean:0:0,s1:1:0,s2:2:0,wrong:0:0:w,none:0:0:n 2>&1 | grep -E "^\[|Traceback|Error"
PENDING="dark cam_front_off cam_front3_off blur extr3 fog radar_off"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ -f $HOME/exp_gcar_b123/pert/$P/DONE ]; then
      echo "== $(date +%H:%M:%S) e60_main on $P"
      ./run_gcar.sh route_head.py --name e60_main --eval_only 1 --eval test --pert $P --conds clean:0:0,s1:1:0,wrong:0:0:w,none:0:0:n 2>&1 | grep -E "^\[|Traceback|Error"
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) done"
