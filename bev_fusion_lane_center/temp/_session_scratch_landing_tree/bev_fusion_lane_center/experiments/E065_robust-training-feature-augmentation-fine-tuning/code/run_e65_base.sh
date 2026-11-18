#!/usr/bin/env bash
# E65: Anpassungs-Kennzahlen des Basis-Kopfes (e37_final) auf den gestoerten TRAIN-Zeilen (steps 0 = nur auswerten)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for P in dark cam_front_off fog; do
  until [ -f $HOME/exp_gcar_b123/pert/$P/DONE_train ]; do sleep 30; done
done
echo "== $(date +%H:%M:%S) base fit"
./run_gcar.sh robust_ft.py --name e65_base --steps 0 --perts dark,cam_front_off,fog 2>&1 | grep -E "^\[|FERTIG|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
