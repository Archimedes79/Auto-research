#!/bin/bash
# run a bev_fusion python script (or -m module) at low priority in the gcar env, from the bev_fusion folder, with `import autogt` working:
#   bash temp/checks/run_py_nice.sh script.py [args]      bash temp/checks/run_py_nice.sh -m autogt selftest
cd "$(dirname "$0")/../.." || exit 1
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python "$@" 2>&1
