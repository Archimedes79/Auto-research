#!/usr/bin/env bash
# Loop 3, E40: Training MIT Pose-Rauschen (1 m / 2 Grad) UND Ausrichtung (Trainings- und Testzeit) - die einsetzbare Konfiguration bei unsicherer Pose.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
if [ ! -f $O/e40_noise_align/metrics.json ]; then
  echo "== $(date +%H:%M:%S) e40_noise_align"
  $G dense_ext.py --name e40_noise_align --init_from e37_final --steps 2000 --lr 5e-4 --bs 8 --workers 3 --eval train_eval,val,test --prior map --prior_xy 1.0 --prior_deg 2 --align 1 2>&1 | grep -E "Eingangs|\[train_eval\]|\[val\]|\[test\]|FERTIG|Error|Traceback" | cut -c1-300
fi
echo "== $(date +%H:%M:%S) e40_noise_align fertig"
