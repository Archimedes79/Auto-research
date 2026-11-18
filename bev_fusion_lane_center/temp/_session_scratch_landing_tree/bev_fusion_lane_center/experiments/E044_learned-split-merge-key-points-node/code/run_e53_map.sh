#!/usr/bin/env bash
# E53 auf das System MIT Karten-Vorwissen (exakte Pose): Verbinder bei Waermeschwelle 0.5 / 0.6, Heat = dichter Kopf e43_joint
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for TH in 0.50 0.60; do
  P="{\"tau\":0.5,\"par\":{\"cut\":1,\"r_s\":16.0,\"r_b\":8.0,\"cos_min\":0.5,\"d_c\":1.5,\"prune\":6.0,\"synth\":$TH}}"
  echo "== $(date +%H:%M:%S) map prior + key points + connectors, tau_h $TH (VAL + TEST)"
  ./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e43_joint_noprop --nodes e44_nodes --heat e43_joint --sets val,test --out e53_map_f$TH --par_json "$P" --workers 6 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
done
echo "== $(date +%H:%M:%S) paths"
./run_gcar.sh path_eval.py e53_map_f0.50/graphs,e53_map_f0.60/graphs --sets val,test 2>&1 | grep -E "Pfad|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
