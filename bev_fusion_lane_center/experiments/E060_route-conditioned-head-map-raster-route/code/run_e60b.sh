#!/usr/bin/env bash
# E60: Kontrollen nacheinander (nach dem Hauptlauf): nur Route (keine Spurkarte), keine Karte
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e60_main/metrics.json ]; do sleep 20; done
./run_gcar.sh route_head.py --name e60_route_only --steps 4000 --maps route > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e60_route_only.out 2>&1
./run_gcar.sh route_head.py --name e60_nomap --steps 4000 --maps none > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e60_nomap.out 2>&1
echo done > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e60b.done
