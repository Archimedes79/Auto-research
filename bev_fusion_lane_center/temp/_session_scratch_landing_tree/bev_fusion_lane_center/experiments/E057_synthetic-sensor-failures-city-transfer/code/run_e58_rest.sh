#!/usr/bin/env bash
# E58: die beiden uebrigen Koepfe nacheinander (nach dem Boston-Kopf)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_B/metrics.json ]; do sleep 20; done
./run_e58.sh S > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_S.out 2>&1
./run_e58.sh M > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_M.out 2>&1
echo "== done" > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e58_rest.done
