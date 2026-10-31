#!/bin/bash
# autogt self-test (nuScenes mini), low priority: bash temp/checks/run_autogt_selftest.sh [more python -m autogt args]
cd "$(dirname "$0")/../.." || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 ~/miniconda3/envs/gcar/bin/python -m autogt "${@:-selftest}" 2>&1
