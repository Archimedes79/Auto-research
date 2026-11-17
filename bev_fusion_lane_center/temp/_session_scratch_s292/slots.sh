#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
S=experiment_execution/runs/.gpu_slots
for d in 0 1; do echo "== slot $d"; ls -la $S/$d 2>&1 | head -5; for f in $S/$d/*; do [ -f "$f" ] && { echo "-- $f"; head -c 400 "$f"; echo; }; done; done
ls $S/.queue 2>/dev/null | head
