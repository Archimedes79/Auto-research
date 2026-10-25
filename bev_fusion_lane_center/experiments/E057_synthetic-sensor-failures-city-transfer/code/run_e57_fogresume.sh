#!/usr/bin/env bash
# E57: fog fortsetzen (zweiter Abbruch durch den Speicherwaechter); nur die leeren Zeilen werden nachgerechnet
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
for TRY in 1 2 3 4; do
  [ -f $HOME/exp_gcar_b123/pert/fog/DONE ] && break
  echo "== $(date +%H:%M:%S) fog Fortsetzung, Versuch $TRY"
  ./run_gcar.sh gcar_pert.py --pert fog --resume 1 --workers 2 2>&1 | grep -E "Fortsetzung|FERTIG|Error|Traceback" | tail -4
done
echo "== $(date +%H:%M:%S) done"
