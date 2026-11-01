#!/bin/bash
# Run a module or a script of bev_fusion at low priority in the gcar env, from the bev_fusion folder, with `import autogt` working
# (one thread, nice 15: the box is shared). From Windows prefix `wsl`:
#   wsl bash autogt/run_nice.sh -m autogt selftest        wsl bash autogt/run_nice.sh my_script.py [args]
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python "$@" 2>&1
