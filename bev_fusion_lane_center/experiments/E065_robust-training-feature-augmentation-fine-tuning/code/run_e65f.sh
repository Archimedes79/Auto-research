#!/usr/bin/env bash
# E65f: Nachtrainieren auf ALLEN sieben Stoerungsarten (echt gestoerte TRAIN-Merkmale), danach Auswertung auf allen TEST-Bedingungen
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for P in dark cam_front_off fog blur cam_front3_off extr3 radar_off; do
  until [ -f $HOME/exp_gcar_b123/pert/$P/DONE_train ]; do sleep 30; done
done
echo "== $(date +%H:%M:%S) fine-tune e65_ftall"
./run_gcar.sh robust_ft.py --name e65_ftall --perts dark,cam_front_off,fog,blur,cam_front3_off,extr3,radar_off --steps 2500 --lr 2e-4 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|Schritt (1000|2000|2500)/"
./run_e65_eval.sh e65_ftall
echo "== $(date +%H:%M:%S) done"
