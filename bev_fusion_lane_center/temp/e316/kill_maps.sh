#!/bin/bash
# E316 kill check: the maps after the joint pair-stage training (runs/e316_p_s<S>_maps) against E307 (runs/e307_n_s<S>): keypoint F1 (3 m, thr 0.3, distance only, mean of entry and exit); kill when more than 0.05 lower. GPU, short, VAL.
cd "$(dirname "$0")/../.."
for S in 0 1; do
  nice bash tools/env/_pyr.sh jmaps_eval.py score --run e316_p_s${S}_maps >> temp/e316/kill_maps_s$S.log 2>&1
done
echo done > temp/e316/kill_maps.done
