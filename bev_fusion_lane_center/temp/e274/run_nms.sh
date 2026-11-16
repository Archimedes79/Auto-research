#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
nice -n 12 bash tools/env/_py14.sh thr_sweep_nms.py e273_p_s0 2>&1 | tail -8
