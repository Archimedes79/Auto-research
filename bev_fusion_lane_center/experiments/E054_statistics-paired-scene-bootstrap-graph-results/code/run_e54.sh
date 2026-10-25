#!/usr/bin/env bash
# E54: Szenen-Bootstrap der kopfzahlen (gepaarte Intervalle)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
echo "== $(date +%H:%M:%S) bootstrap start"
./run_gcar.sh boot_eval.py --sets val,test --draws 2000 --workers 5 2>&1 | grep -v Warning
echo "== $(date +%H:%M:%S) done"
