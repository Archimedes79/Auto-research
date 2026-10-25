#!/usr/bin/env bash
# E65b: gestoerte TRAIN-Merkmale (jede 10. Trainingszeile); startet erst, wenn die Auswertungsjobs (E58, E61, E62) fertig sind (Speicher: jeder Extraktions-Arbeiter haelt eine Kopie der nuScenes-Tabellen)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e61[.]sh|run_e62[.]sh|run_e58_train[.]sh|run_e58_retry[.]sh|run_e59b[.]sh|gcar_pert[.]py --pert radar_off" > /dev/null; do sleep 30; done
for P in dark cam_front_off fog; do
  rm -f $HOME/exp_gcar_b123/pert/$P/DONE_train
  echo "== $(date +%H:%M:%S) train $P"
  ./run_gcar.sh gcar_pert.py --pert $P --part train --stride 10 --workers 2 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
done
echo "== $(date +%H:%M:%S) done"
