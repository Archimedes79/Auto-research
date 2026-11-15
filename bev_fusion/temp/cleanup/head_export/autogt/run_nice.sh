#!/bin/bash
# Run a module or a script at low priority in the gcar env with `import autogt` working (one thread, nice 15: the box is shared).
# The working directory stays the caller's. From Windows prefix `wsl`, from the repository root:
#   wsl bash autogt/run_nice.sh -m autogt selftest        wsl bash autogt/run_nice.sh bev_fusion/tests/test_orientation.py [args]
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python "$@" 2>&1
