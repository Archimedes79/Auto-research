#!/usr/bin/env bash
# E62: Planer-nahe Auswertung (planning-lite): Bedingungen der Zuordnung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
R="./run_gcar.sh plan_eval.py --sets val,test --workers 4"
$R --sigma 0 --reg 0 --out e62_s0 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
$R --sigma 1.0 --reg 1 --out e62_s1_reg 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
$R --sigma 1.0 --reg 0 --out e62_s1_noreg 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
$R --sigma 2.0 --reg 1 --out e62_s2_reg 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
$R --drop 0.3 --protect 1 --reg 1 --out e62_d03 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
echo "== done"
