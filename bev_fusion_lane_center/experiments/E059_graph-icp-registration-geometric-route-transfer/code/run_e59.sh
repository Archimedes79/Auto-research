#!/usr/bin/env bash
# E59: Zuordnung + Routen-Uebertragung: Bedingungs-Gitter (TEST + VAL)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
R="./run_gcar.sh assoc_eval.py --sets val,test --workers 8"
echo "== $(date +%H:%M:%S) oracle / exact pose / no map"
$R --oracle 1 --out e59_oracle 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --reg 0 --out e59_s0 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --nomap 1 --out e59_nomap 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
for S in 0.5 1.0 2.0 3.0; do
  echo "== $(date +%H:%M:%S) pose noise $S"
  $R --sigma $S --reg 0 --out e59_s${S}_noreg 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
  $R --sigma $S --reg 1 --out e59_s${S}_reg 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
done
for D in 0.1 0.3 0.5; do
  echo "== $(date +%H:%M:%S) outdated map: drop $D (route kept / route may vanish)"
  $R --drop $D --protect 1 --reg 1 --out e59_d${D}_keep 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
  $R --drop $D --protect 0 --reg 1 --out e59_d${D}_any 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
done
echo "== $(date +%H:%M:%S) re-painted lanes (30 % shifted 1.5 m) and combined"
$R --shift_frac 0.3 --reg 1 --out e59_shift30 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$R --sigma 1.0 --drop 0.2 --protect 1 --reg 1 --out e59_s1_d02 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) done"
