#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh temp/e372/pos_grad.py --n 8 > temp/e372/pos_grad.out 2>&1
