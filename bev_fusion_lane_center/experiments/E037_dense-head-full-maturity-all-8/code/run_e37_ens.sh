#!/usr/bin/env bash
# Loop 3, E37: Mittel der vier dichten Koepfe (Endkopf + drei Kreuz-Anpassungen) auf VAL und TEST (nur Auswertung).
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
until [ -f $O/e37_final/metrics.json ] && [ -f $O/e37_fold0/metrics.json ] && [ -f $O/e37_fold1/metrics.json ] && [ -f $O/e37_fold2/metrics.json ]; do sleep 30; done
echo "== $(date +%H:%M:%S) alle dichten Koepfe fertig -> Mittel"
/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh make_ensemble.py e37_final,e37_fold0,e37_fold1,e37_fold2 e37_ens 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-300
echo "== $(date +%H:%M:%S) Mittel fertig"
