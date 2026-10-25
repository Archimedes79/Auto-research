#!/usr/bin/env bash
# E64 (Erweiterung): Monitor gegen unbekannte Ausfallarten (leave-one-condition-out) -> out/loop2/e64_monitors/loco.json
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
./run_gcar.sh monitor_loco.py 2>&1 | tail -n 8
./run_gcar.sh monitor_scenecv.py 2>&1 | tail -n 4
