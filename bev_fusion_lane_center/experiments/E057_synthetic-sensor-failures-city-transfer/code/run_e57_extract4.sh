#!/usr/bin/env bash
# E57: radar_off (TEST) neu extrahieren (der Lauf nach extr3 scheiterte an einem Schreibfehler unter Speichermangel); belegt den Platz des Strangs C (fog) nach dessen Ende
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e57_extract3[.]sh" > /dev/null; do sleep 20; done
rm -f $HOME/exp_gcar_b123/pert/radar_off/DONE
echo "== $(date +%H:%M:%S) radar_off (TEST, zweiter Versuch)"
./run_gcar.sh gcar_pert.py --pert radar_off --workers 2 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
echo "== $(date +%H:%M:%S) done"
