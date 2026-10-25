#!/usr/bin/env bash
# E59 (Erweiterung): Routen-Uebertragung auf den gestoerten Graphen von E57 (TEST, Pose-Rauschen 1 m, Einzelbild-Registrierung)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for C in stored dark cam_front3_off blur cam_front_off fog radar_off extr3; do
  ./run_gcar.sh assoc_eval.py --sets test --graphs e57_$C/graphs --sigma 1.0 --reg 1 --workers 10 --out e59p_$C 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-400
done
echo "== $(date +%H:%M:%S) done"
