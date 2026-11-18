#!/usr/bin/env bash
# Loop 3, E43: weitere Gesamtsystem-Laeufe (zeitliche Variante A, Karte mit exakter Pose, Karte mit Rauschen + Ausrichtung), jeweils mit dem Verfeinerer e39_main und gelernten Verbindungen.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
until grep -q "E43 Basis fertig (zweiter Anlauf)" $O/stageB2.out 2>/dev/null; do sleep 20; done
TL=$($G pick_tau_l.py | tail -n 1)
echo "== $(date +%H:%M:%S) E43c: tau_l = $TL"
for r in e41_A e40_clean e40_noise_align; do
  until [ -f $O/$r/metrics.json ]; do sleep 20; done
  $G final_system.py --dense $r --no_prop --tau_l $TL --out e43 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-330
done
echo "== $(date +%H:%M:%S) E43c fertig"
