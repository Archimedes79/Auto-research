#!/usr/bin/env bash
# E59: Viterbi-Zuordnung laengs der Polylinien (Kartenstuecke als Zustaende, Nachfolger-Kanten als Uebergaenge)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
R="./run_gcar.sh assoc_eval.py --sets val,test --workers 6 --hmm 1"
$R --reg 0 --out e59_s0_hmm 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --sigma 1.0 --reg 1 --out e59_s1.0_reg_hmm 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --sigma 2.0 --reg 1 --out e59_s2.0_reg_hmm 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --drop 0.3 --protect 1 --reg 1 --out e59_d0.3_keep_hmm 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== done"
