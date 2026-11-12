#!/bin/bash
# pictures and videos of a trained E273 run on three VAL scenes (CPU, 1 thread): $1 = run name, $2 = launcher id
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
nice -n 12 bash tools/env/_py14.sh qgraph_results.py --run $1 --id $2 --device cpu --scenes 11 39 23 2>&1 | tail -5
