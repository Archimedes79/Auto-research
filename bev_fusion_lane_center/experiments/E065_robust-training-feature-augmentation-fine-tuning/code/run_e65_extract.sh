#!/usr/bin/env bash
# E65b: gestoerte TRAIN-Merkmale (jede 10. Trainingszeile) fuer das robuste Nachtrainieren; belegt den Platz des Strangs B nach dessen Ende
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e57_extract2.sh dark cam_front3_off" > /dev/null; do sleep 20; done
for P in dark cam_front_off fog; do
  rm -f $HOME/exp_gcar_b123/pert/$P/DONE_train
  echo "== $(date +%H:%M:%S) train $P"
  ./run_gcar.sh gcar_pert.py --pert $P --part train --stride 10 --workers 3 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
done
echo "== $(date +%H:%M:%S) done"
