#!/usr/bin/env bash
# E56: gewaehlte mittlere Einstellung (heat 0.5, Richtungs-Kosinus 0.6; vorab festgelegt, nicht auf TEST abgestimmt) auf VAL + TEST
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
P='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0,"synth":0.5,"cos_c":0.6}}'
echo "== $(date +%H:%M:%S) synth tau_h 0.5 cos_c 0.6 (VAL + TEST)"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --heat e41_A --sets val,test --out e56_mid --par_json "$P" --workers 4 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
./run_gcar.sh path_eval.py e56_mid/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
