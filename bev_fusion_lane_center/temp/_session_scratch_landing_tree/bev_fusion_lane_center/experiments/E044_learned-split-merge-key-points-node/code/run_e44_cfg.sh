#!/usr/bin/env bash
# E44: feste Konfigurationen des Knoten-gefuehrten Aufbaus auf VAL / TEST: ausgewogen (Route F-Verlust <= 0,015), Obergrenze mit Soll-Knoten statt Schluesselpunkten
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
echo "== $(date +%H:%M:%S) balanced"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e44_balanced --par_json '{"tau":0.5,"par":{"snap":8.0,"create":1,"r_b":6.0,"cos_min":0.5,"prune":8.0}}' --workers 6 2>&1 | grep -E "^\[|Traceback|Error"
echo "== $(date +%H:%M:%S) oracle (GT nodes as key points, aggressive configuration)"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e44_oracle --oracle 1 --par_json '{"tau":0.5,"par":{"snap":12.0,"create":1,"r_b":16.0,"cos_min":0.2,"prune":6.0}}' --workers 6 2>&1 | grep -E "^\[|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
