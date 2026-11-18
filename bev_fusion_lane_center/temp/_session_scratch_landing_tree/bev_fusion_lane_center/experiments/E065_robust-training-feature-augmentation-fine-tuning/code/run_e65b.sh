#!/usr/bin/env bash
# E65b/c: Nachtrainieren auf echt gestoerten TRAIN-Merkmalen (nachdem gcar_pert.py --part train fuer dark, cam_front_off, fog fertig ist), danach Auswertung auf allen TEST-Bedingungen; E65c = zusaetzlich Merkmalsraum-Augmentierung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for P in dark cam_front_off fog; do
  until [ -f $HOME/exp_gcar_b123/pert/$P/DONE_train ]; do sleep 30; done
done
echo "== $(date +%H:%M:%S) fine-tune e65_ft"
./run_gcar.sh robust_ft.py --name e65_ft --perts dark,cam_front_off,fog --steps 2000 --lr 2e-4 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|Schritt (1000|2000)/"
./run_e65_eval.sh e65_ft
echo "== $(date +%H:%M:%S) fine-tune e65_ftaug"
./run_gcar.sh robust_ft.py --name e65_ftaug --perts dark,cam_front_off,fog --steps 2000 --lr 2e-4 --faug 0.25 2>&1 | grep -E "^\[|FERTIG|Traceback|Error|Schritt (1000|2000)/"
./run_e65_eval.sh e65_ftaug
echo "== $(date +%H:%M:%S) done"
