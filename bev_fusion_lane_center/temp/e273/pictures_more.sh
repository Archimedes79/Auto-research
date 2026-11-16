#!/bin/bash
# more VAL scenes for the report (CPU, 1 thread): 27 = scene-0096 and 17 = scene-0017 (busy intersections), 15 = scene-0015 (narrow road, bus)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
for s in 0 1; do
  nice -n 12 bash tools/env/_py14.sh qgraph_results.py --run e273_p_s$s --id E273.p_s$s --device cpu --scenes 27 17 15 2>&1 | tail -2
done
