#!/usr/bin/env bash
# E62: Wiederholungslauf - jede Bedingung bis zu 4 Mal, bis metrics.json den Testteil mit den verankerten Kennzahlen (lata) enthaelt (unter Speichermangel scheitert fork gelegentlich)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e62[.]sh" > /dev/null; do sleep 20; done
run() {  # name args...
  N=$1; shift
  for TRY in 1 2 3 4; do
    if grep -q '"test"' /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/$N/metrics.json 2>/dev/null && grep -q lata /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/$N/metrics.json 2>/dev/null; then return 0; fi
    echo "== $(date +%H:%M:%S) $N try $TRY"
    ./run_gcar.sh plan_eval.py --sets val,test --workers 3 "$@" --out $N 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-420
  done
}
run e62_s0 --sigma 0 --reg 0
run e62_s1_reg --sigma 1.0 --reg 1
run e62_s1_noreg --sigma 1.0 --reg 0
run e62_s2_reg --sigma 2.0 --reg 1
run e62_d03 --drop 0.3 --protect 1 --reg 1
echo "== $(date +%H:%M:%S) done"
