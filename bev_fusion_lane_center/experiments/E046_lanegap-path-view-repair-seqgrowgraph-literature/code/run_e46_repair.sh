#!/usr/bin/env bash
# E46: Kontinuitaets-Reparatur (innere Sackgassen / Waisen mit passendem Stueck verbinden) ohne Schluesselpunkte; danach Pfad-Auswertung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e46_repair --par_json '{"tau":0.5,"par":{"snap":0,"create":0,"prune":0,"repair":12.0,"rep_cos":0.7}}' --workers 4 2>&1 | grep -E "^\[|Traceback|Error"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e46_repair2 --par_json '{"tau":0.5,"par":{"snap":0,"create":0,"prune":0,"repair":20.0,"rep_cos":0.4}}' --workers 4 2>&1 | grep -E "^\[|Traceback|Error"
./run_gcar.sh path_eval.py e46_repair/graphs,e46_repair2/graphs,e44_balanced/graphs,e44_oracle/graphs,e43/graphs_rule_e43_joint_noprop --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error"
echo "== $(date +%H:%M:%S) e46 repair done"
