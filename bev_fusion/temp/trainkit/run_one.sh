#!/usr/bin/env bash
# scratch: run one python script of bev_fusion in the gcar env with the repo root on PYTHONPATH (2 threads); usage: run_one.sh <script> [args]
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$HOME/miniconda3/envs/gcar/bin/python" "$@"
