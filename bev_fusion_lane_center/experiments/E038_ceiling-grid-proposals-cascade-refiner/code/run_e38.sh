#!/usr/bin/env bash
# Loop 3, E38: Obergrenze der Gitter-Vorschlaege des dichten Endkopfes fuer alle drei Verfolger-Schwellen (strict / relax / wide) auf VAL und TEST.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
until [ -f $O/e37_final/metrics.json ]; do sleep 20; done
echo "== $(date +%H:%M:%S) E38 Obergrenzen"
for tag in strict wide; do for st in val test; do $G proposals.py --name e37_final --set $st --tag $tag --ceiling --workers 3 2>&1 | tail -2 | cut -c1-300; done; done
echo "== $(date +%H:%M:%S) E38 fertig"
