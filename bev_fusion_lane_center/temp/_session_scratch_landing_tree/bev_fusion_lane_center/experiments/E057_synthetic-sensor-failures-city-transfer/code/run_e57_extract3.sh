#!/usr/bin/env bash
# E57: fog neu extrahieren (der erste Lauf wurde vom Speicherwaechter beendet); wartet auf das Ende des Strangs A (dark fog radar_off blur ...)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
while pgrep -f "run_e57_extract2.sh dark fog" > /dev/null; do sleep 20; done
rm -f $HOME/exp_gcar_b123/pert/fog/DONE
echo "== $(date +%H:%M:%S) fog (neu)"
./run_gcar.sh gcar_pert.py --pert fog --workers 3 2>&1 | grep -E "FERTIG|Error|Traceback" | tail -3
echo "== $(date +%H:%M:%S) done"
