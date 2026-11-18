#!/bin/bash
# E317 scratch: the real-frame tests (CPU, 2 threads).  usage: run_real.sh <regress | geom | bench | flops>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
case "$1" in
  regress)
    for v in orig dev; do bash tools/env/_py14.sh e320/t_real.py fwd $v || exit 1; done
    bash tools/env/_py14.sh e320/t_real.py compare ;;
  *) bash tools/env/_py14.sh e320/t_real.py "$1" ;;
esac
