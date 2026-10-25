#!/usr/bin/env bash
# Loop 3, E39 Anpassung gegen die Luecke Trainings-/Validierungsszenen: Kanal-Ausfall und Rauschen auf den Grundlage-Merkmalen im Training des Verfeinerers.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
TR="e37_fold0:oof:train,e37_fold1:oof:train,e37_fold2:oof:train"
EV="val=e37_final:val:val,test=e37_final:test:val,oofA=e37_fold0:oof:train"
if [ ! -f $O/e39_reg/metrics.json ]; then
  echo "== $(date +%H:%M:%S) e39_reg"
  $G refiner.py --name e39_reg --workers 4 --train $TR --eval $EV --steps 5000 --blocks 3 --lam_fp 3 --jitter 0.5 --in_drop 0.2 --in_noise 0.3 2>&1 | grep -E "Parameter|Trainings|\[val\]|\[test\]|\[oofA\]|FERTIG|Error|Traceback" | cut -c1-330
fi
echo "== $(date +%H:%M:%S) e39_reg fertig"
