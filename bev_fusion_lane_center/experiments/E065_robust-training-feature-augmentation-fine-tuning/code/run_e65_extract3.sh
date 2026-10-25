#!/usr/bin/env bash
# E65b: gestoerte TRAIN-Merkmale parallel in zwei Straengen (je 2 Arbeiter): Strang 1 = cam_front_off nach dark, Strang 2 = fog nach dem Ende der fog-TEST-Extraktion
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
P=$1
case "$P" in
  cam_front_off) until [ -f $HOME/exp_gcar_b123/pert/dark/DONE_train ]; do sleep 30; done ;;
  fog) until [ -f $HOME/exp_gcar_b123/pert/fog/DONE ]; do sleep 30; done ;;
esac
rm -f $HOME/exp_gcar_b123/pert/$P/DONE_train
echo "== $(date +%H:%M:%S) train $P"
./run_gcar.sh gcar_pert.py --pert $P --part train --stride 10 --workers 2 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
echo "== $(date +%H:%M:%S) done"
