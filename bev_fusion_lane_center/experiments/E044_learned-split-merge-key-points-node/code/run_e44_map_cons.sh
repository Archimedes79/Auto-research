#!/usr/bin/env bash
# E44: Standard (konservativer Schnitt + Entfernen) auch auf das System MIT Karten-Vorwissen; Pfad-Auswertung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
C2='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0}}'
echo "== $(date +%H:%M:%S) conservative cut + prune on the map-prior system"
./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e43_joint_noprop --nodes e44_nodes --sets val,test --out e44_cons_map --par_json "$C2" --workers 4 2>&1 | grep -E "^\[(val|test)\]|Traceback|Error"
./run_gcar.sh path_eval.py e44_cons_map/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
