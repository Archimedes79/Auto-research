#!/usr/bin/env bash
# E73: gestoerte Merkmale (7 Stoerungen) auch fuer die VAL-Szenen (jede 3. VAL-Zeile), damit die Wahl der Feinabstimmung ohne TEST erfolgt.
# Aufruf: run_e73_extract.sh A|B   (zwei Ketten parallel)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
CHAIN=$1
if [ "$CHAIN" = "A" ]; then LIST="dark cam_front_off fog radar_off"; else LIST="cam_front3_off blur extr3"; fi
for P in $LIST; do
  echo "== $(date +%H:%M:%S) val $P"
  ./run_gcar.sh gcar_pert.py --pert $P --part val --stride 3 --workers 3 2>&1 | grep -E "^[a-z_0-9]+ [0-9]+/|FERTIG|Traceback|Error|Wiederholung" | tail -n 8
done
echo "== $(date +%H:%M:%S) done $CHAIN"
