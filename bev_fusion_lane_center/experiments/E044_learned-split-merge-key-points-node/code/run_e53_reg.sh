#!/usr/bin/env bash
# E53 / E48: Knotenschritte (Standard-Schnitt + Entfernen, Verbinder) auf dem System MIT registriertem, verrauschtem Karten-Vorwissen (sigma 1 m, Korrelation + gelernte Feinstufe)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
echo "== $(date +%H:%M:%S) final system on the registered prior (dense run e48_clean_s1_corrfine)"
./run_gcar.sh final_system.py --dense e48_clean_s1_corrfine --no_prop --tau_l 0.3 --sets val,test --out e53_regsys 2>&1 | grep -E "^\[|Traceback|Error" | cut -c1-300
ls /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e53_regsys
G=e53_regsys/graphs_rule_e48_clean_s1_corrfine_noprop
C2='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0}}'
C3='{"tau":0.5,"par":{"cut":1,"r_s":16.0,"r_b":8.0,"cos_min":0.5,"d_c":1.5,"prune":6.0,"synth":0.5}}'
echo "== $(date +%H:%M:%S) default key-point nodes"
./run_gcar.sh node_assemble.py --graphs $G --nodes e44_nodes --heat e48_clean_s1_corrfine --sets val,test --out e53_reg_cons --par_json "$C2" --workers 6 2>&1 | grep -E "^\[(val|test)\]|Traceback|Error" | cut -c1-330
echo "== $(date +%H:%M:%S) + connectors heat 0.5"
./run_gcar.sh node_assemble.py --graphs $G --nodes e44_nodes --heat e48_clean_s1_corrfine --sets val,test --out e53_reg_conn50 --par_json "$C3" --workers 6 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error" | cut -c1-330
./run_gcar.sh path_eval.py e53_reg_cons/graphs,e53_reg_conn50/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error" | cut -c1-250
echo "== $(date +%H:%M:%S) done"
