#!/usr/bin/env bash
# E61: zeitliche Verfolgung der Kartenausrichtung (Token-Markierung + geometrische Routen-Markierung); wartet auf das Ende des E59/E62-Laufs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
export NO_SLOT=1
if [ "$1" = "wait" ]; then
  while ! grep -q "ALL done" ../out/loop2/e59_grid2.out; do sleep 30; done
fi
for S in 1 2 3; do
  ./run_gcar.sh assoc_temporal.py --sigma $S.0 --out e61_s$S --workers 6 2>&1 | grep -E "^\[|Error|Traceback" | cut -c1-1500 > ../out/loop2/e61_s$S.out
done
echo "== $(date +%H:%M:%S) E61 done" >> ../out/loop2/e61_done.out
