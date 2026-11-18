#!/usr/bin/env bash
# E69: Karten-Spuren (mit den Registrierungen aus E67) als Spurgraph waehrend des Ausfalls; benoetigt out/loop2/e67_s1/frames_*.pkl (hold_eval.py neuer Stand mit reg / noise)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
./run_gcar.sh maplane_eval.py --workers 10 --out e69_maplanes 2>&1 | grep -E "^\[|fertig|Traceback|Error"
./run_gcar.sh stale_signal.py 2>&1 | tail -n 3
./run_gcar.sh policy_eval.py 2>&1 | tail -n 5
