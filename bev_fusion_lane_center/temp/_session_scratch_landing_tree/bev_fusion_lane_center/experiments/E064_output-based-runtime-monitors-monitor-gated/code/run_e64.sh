#!/usr/bin/env bash
# E64: Monitore - rechnet, sobald die E57-Graphen und die E62p-Planerauswertungen aller Bedingungen vorliegen (oder nach dem Zeitlimit mit den vorhandenen)
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
CONDS="stored dark cam_front_off cam_front3_off blur extr3 fog radar_off"
LIMIT=${1:-0}   # Sekunden Wartezeit, 0 = unbegrenzt
T0=$(date +%s)
while true; do
  ALL=1
  for C in $CONDS; do [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e62p_$C/persample_test.pkl ] || ALL=0; done
  [ $ALL = 1 ] && break
  if [ $LIMIT -gt 0 ] && [ $(( $(date +%s) - T0 )) -gt $LIMIT ]; then break; fi
  sleep 60
done
HAVE=""
for C in $CONDS; do [ -f /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e62p_$C/persample_test.pkl ] && HAVE="$HAVE,$C"; done
HAVE=${HAVE#,}
echo "== $(date +%H:%M:%S) monitor_eval on $HAVE"
./run_gcar.sh monitor_eval.py --conds $HAVE --workers 4 2>&1 | grep -E "^\[|pooled|detect|Traceback|Error|uebersprungen"
echo "== $(date +%H:%M:%S) done"
