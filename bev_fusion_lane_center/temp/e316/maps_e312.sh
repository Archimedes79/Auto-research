#!/bin/bash
# reference for the E316 kill check: the maps after the joint pair training of E312 (3000 steps), keypoint F1 on VAL. GPU, short.
cd "$(dirname "$0")/../.."
for S in 0 1; do
  nice bash tools/env/_pyr.sh jmaps_eval.py score --run e312_p_s${S}_maps >> temp/e316/maps_e312_s$S.log 2>&1
done
echo done > temp/e316/maps_e312.done
