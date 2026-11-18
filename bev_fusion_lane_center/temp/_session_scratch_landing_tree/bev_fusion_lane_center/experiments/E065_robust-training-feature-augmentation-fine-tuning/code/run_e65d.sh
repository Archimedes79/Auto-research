#!/usr/bin/env bash
# E65d: routen-bedingter Kopf (E60) auf echt gestoerten TRAIN-Merkmalen nachtrainiert: faellt er bei Sensorausfall auf die Karte zurueck, ohne einer falschen Karte zu trauen?
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e65_ft/model.pt ]; do sleep 30; done
echo "== $(date +%H:%M:%S) fine-tune e65d_route_ft"
./run_gcar.sh route_head.py --name e65d_route_ft --init_from e60_main --steps 1500 --lr 2e-4 --pert_train dark,cam_front_off,fog --eval test --workers 3 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|ROLLBACK|E65d|Schritt (500|1000|1500)/"
PENDING="dark cam_front_off fog cam_front3_off blur extr3 radar_off"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ -f $HOME/exp_gcar_b123/pert/$P/DONE ]; then
      echo "== $(date +%H:%M:%S) e65d_route_ft on $P"
      ./run_gcar.sh route_head.py --name e65d_route_ft --eval_only 1 --eval test --pert $P --conds clean:0:0,s1:1:0,wrong:0:0:w,none:0:0:n 2>&1 | grep -E "^\[|Traceback|Error"
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) done"
