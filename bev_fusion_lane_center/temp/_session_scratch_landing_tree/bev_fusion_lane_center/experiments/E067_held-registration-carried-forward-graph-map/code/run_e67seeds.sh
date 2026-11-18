#!/usr/bin/env bash
# E67: Seed-Wiederholung der Pose-Fehler-Realisierung (Seeds 1-4; Seed 0 = Hauptlauf e67_s1)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for SD in 1 2 3 4; do
  ./run_gcar.sh hold_eval.py --sigma 1.0 --seed $SD --workers 8 --out e67_seed$SD 2>&1 | grep -E "Mittel|Traceback|Error"
done
echo "== $(date +%H:%M:%S) done"
