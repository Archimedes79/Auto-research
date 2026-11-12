#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
bash tools/env/_py14.sh qgraph_results.py --run e273_p_s0 --id l6_view_B --scenes 11 --every 2 --device cpu
