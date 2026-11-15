#!/bin/bash
# full scoring of a trained E273 run on the 873 VAL frames (GPU, shared with a running training)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
nice -n 8 bash tools/env/_py14.sh qgraph_run_eval.py score --run $1 --seed $2 2>&1 | tail -${3:-40}
