#!/usr/bin/env bash
# E65e: AdaBN (unueberwachte BN-Anpassung zur Testzeit) - Bedingungen in der Reihenfolge ihrer Fertigstellung
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
PENDING="stored dark cam_front_off cam_front3_off blur extr3 radar_off fog"
while [ -n "$PENDING" ]; do
  NEXT=""
  for P in $PENDING; do
    if [ "$P" = "stored" ] || [ -f $HOME/exp_gcar_b123/pert/$P/DONE ]; then
      echo "== $(date +%H:%M:%S) adabn $P"
      ./run_gcar.sh eval_adabn.py --perts $P --workers 3 2>&1 | grep -E "^\[|Traceback|Error"
    else
      NEXT="$NEXT $P"
    fi
  done
  PENDING="$NEXT"
  [ -n "$PENDING" ] && sleep 30
done
echo "== $(date +%H:%M:%S) done"
