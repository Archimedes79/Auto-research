#!/usr/bin/env bash
# Loop 3, E41 Variante B (Weitergabe der Anfragen): Auswertung auf VAL, TEST und (Trainings-Szenen) OOF-Fold 0 mit dem Verfeinerer e39_main.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
until [ -f $O/e39_main/metrics.json ] && [ -f $O/e37_final/metrics.json ]; do sleep 30; done
echo "== $(date +%H:%M:%S) E41 B"
$G temporal_b.py --refiner e39_main --run e37_final --set val --part val --name e41_B --workers 3 2>&1 | grep -E "^\[|Wellen|Error|Traceback" | cut -c1-330
$G temporal_b.py --refiner e39_main --run e37_final --set test --part val --name e41_B --workers 3 2>&1 | grep -E "^\[|Wellen|Error|Traceback" | cut -c1-330
$G temporal_b.py --refiner e39_main --run e37_fold0 --set oof --part train --name e41_B_oof --workers 3 --variants gate:0.6:0 2>&1 | grep -E "^\[|Wellen|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) E41 B fertig"
