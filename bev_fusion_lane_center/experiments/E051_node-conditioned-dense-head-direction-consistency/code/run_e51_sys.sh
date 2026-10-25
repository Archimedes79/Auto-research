#!/usr/bin/env bash
# E51: Systemwirkung des Knoten-konditionierten dichten Kopfes (e51_A_nodes): Vorschlaege -> Verfeinerer e39_main -> Regel-Verbindungen (final_system.py); danach Knoten-Einbau (konservativ + Entfernen) und Pfad-Auswertung
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
export NO_SLOT=1
until [ -f $O/e51_A_nodes/metrics.json ]; do sleep 15; done
echo "== $(date +%H:%M:%S) final_system e51"
$G final_system.py --dense e51_A_nodes --no_prop --links "" --out e51 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) key-point nodes on the e51 graphs"
C2='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0}}'
$G node_assemble.py --graphs e51/graphs_rule_e51_A_nodes_noprop --nodes e44_nodes --sets val,test --out e51_cons --par_json "$C2" --workers 4 2>&1 | grep -E "^\[(val|test)\]|Traceback|Error" | cut -c1-330
$G path_eval.py e51/graphs_rule_e51_A_nodes_noprop,e51_cons/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error" | cut -c1-300
echo "== $(date +%H:%M:%S) e51 sys done"
