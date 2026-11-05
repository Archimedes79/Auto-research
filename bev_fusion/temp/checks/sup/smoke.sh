#!/bin/bash
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
~/miniconda3/envs/gcar/bin/python tests/test_flow_timed.py --boxes --maps --quick --iters 20 > temp/checks/sup/smoke.log 2>&1
echo "exit $?" >> temp/checks/sup/smoke.log
