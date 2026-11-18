#!/usr/bin/env bash
# Loop 3, E43: Gesamtsystem-Laeufe (Einzelbild mit Regel- und gelernten Verbindungen; mit Weitergabe). Wartet auf Stufe B (Verbindungskopf) und den Endkopf.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
until grep -q "Stufe B fertig" $O/stageB.out 2>/dev/null; do sleep 30; done
TL=$($G pick_tau_l.py | tail -n 1)
echo "== $(date +%H:%M:%S) E43: tau_l (VAL) = $TL"
$G final_system.py --dense e37_final --no_prop --tau_l $TL --out e43 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
$G final_system.py --dense e37_final --gate 0.6 --tau_l $TL --out e43 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
echo "== $(date +%H:%M:%S) E43 Basis fertig"
