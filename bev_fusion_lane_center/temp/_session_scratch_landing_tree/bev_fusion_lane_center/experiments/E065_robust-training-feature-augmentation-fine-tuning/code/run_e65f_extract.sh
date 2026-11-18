#!/usr/bin/env bash
# E65f: gestoerte TRAIN-Merkmale der uebrigen vier Stoerungen (jede 10. Trainingszeile) in zwei Straengen mit je 2 Arbeitern: Strang A = blur, extr3; Strang B = cam_front3_off, radar_off
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
S=$1; shift
for P in "$@"; do
  rm -f $HOME/exp_gcar_b123/pert/$P/DONE_train
  echo "== $(date +%H:%M:%S) train $P"
  for TRY in 1 2 3; do
    [ -f $HOME/exp_gcar_b123/pert/$P/DONE_train ] && break
    ./run_gcar.sh gcar_pert.py --pert $P --part train --stride 10 --workers 2 $( [ $TRY -gt 1 ] && echo "--resume 1" ) 2>&1 | grep -E "Fortsetzung|FERTIG|Error|Traceback" | tail -3
  done
done
echo "== $(date +%H:%M:%S) done $S"
