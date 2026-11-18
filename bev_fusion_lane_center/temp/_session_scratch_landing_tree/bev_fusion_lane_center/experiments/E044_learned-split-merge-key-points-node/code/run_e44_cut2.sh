#!/usr/bin/env bash
# E44: geometrieerhaltender Knoteneinbau (cut): ohne Entfernen (prune), Obergrenze mit Soll-Knoten, Anwendung auf das System MIT Karten-Vorwissen; danach Pfad-Auswertung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
CUT='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":16.0,"cos_min":0.0,"d_c":1.5,"prune":6.0}}'
CUTNP='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":16.0,"cos_min":0.0,"d_c":1.5}}'
echo "== $(date +%H:%M:%S) cut without prune"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e44_cut_np --par_json "$CUTNP" --workers 4 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
echo "== $(date +%H:%M:%S) cut with GT nodes (oracle)"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --out e44_cut_oracle --oracle 1 --par_json "$CUT" --workers 4 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
echo "== $(date +%H:%M:%S) cut on the map-prior system"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e43_joint_noprop --nodes e44_nodes --sets val,test --out e44_cut_map --par_json "$CUT" --workers 4 2>&1 | grep -E "^\[(val|test)\]|Traceback|Error"
echo "== $(date +%H:%M:%S) paths"
./run_gcar.sh path_eval.py e44_cut/graphs,e44_cut_np/graphs,e44_cut_map/graphs,e44_cut_oracle/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
