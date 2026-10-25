#!/usr/bin/env bash
# E53: hoehere Waerme-Schwellen der Verbinder-Erzeugung (VAL)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for TH in 0.50 0.70; do
  P="{\"tau\":0.5,\"par\":{\"cut\":1,\"r_s\":16.0,\"r_b\":8.0,\"cos_min\":0.5,\"d_c\":1.5,\"prune\":6.0,\"synth\":$TH}}"
  echo "== $(date +%H:%M:%S) synth tau_h $TH (VAL)"
  ./run_gcar.sh node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --heat e41_A --sets val --out e53_t$TH --par_json "$P" --workers 4 2>&1 | grep -E "^\[(val|test)\] \{|Traceback|Error"
done
echo "== $(date +%H:%M:%S) done"
