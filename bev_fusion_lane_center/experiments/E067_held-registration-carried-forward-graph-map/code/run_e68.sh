#!/usr/bin/env bash
# E68: Ausfall ueberbruecken - zuletzt guten Graphen per Eigenbewegung weitergeben. Hauptlauf (exakte Eigenbewegung) und mit Kilometerzaehler-Fehler
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
./run_gcar.sh coast_eval.py --workers 4 --out e68_main 2>&1 | grep -E "^\[|Mittel|Traceback|Error"
./run_gcar.sh coast_eval.py --workers 4 --odo 1.0 --out e68_odo 2>&1 | grep -E "^\[|Mittel|Traceback|Error"
./run_gcar.sh coast_eval.py --workers 6 --scenecv 1 --out e68_scv 2>&1 | grep -E "^\[|Mittel|Traceback|Error"
echo "== $(date +%H:%M:%S) done"
