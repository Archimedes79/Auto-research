#!/usr/bin/env bash
# E57: extr3 fortsetzen (der Lauf wurde bei Zeile ~405 von 417 vom Speicherwaechter beendet); wartet auf das Ende von radar_off
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
until [ -f $HOME/exp_gcar_b123/pert/radar_off/DONE ]; do sleep 20; done
echo "== $(date +%H:%M:%S) extr3 (Fortsetzung)"
./run_gcar.sh gcar_pert.py --pert extr3 --resume 1 --workers 1 2>&1 | grep -E "Fortsetzung|FERTIG|Error|Traceback" | tail -4
echo "== $(date +%H:%M:%S) done"
