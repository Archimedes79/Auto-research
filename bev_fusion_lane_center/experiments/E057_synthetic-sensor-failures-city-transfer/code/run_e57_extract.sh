#!/usr/bin/env bash
# E57: Merkmale unter Sensorstoerungen auf den TEST-Zeilen extrahieren (zwei parallele Straenge)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
STREAM=$1
if [ "$STREAM" = "A" ]; then LIST="dark fog blur noise night_radar extr1"; else LIST="cam_front_off cam_front3_off radar_drop90 radar_off extr3 clean"; fi
for P in $LIST; do
  echo "== $(date +%H:%M:%S) $P"
  ./run_gcar.sh gcar_pert.py --pert $P --workers 3 2>&1 | grep -E "FERTIG|Error|Traceback|^$P [0-9]+/833 " | tail -3
done
echo "== $(date +%H:%M:%S) stream $STREAM done"
