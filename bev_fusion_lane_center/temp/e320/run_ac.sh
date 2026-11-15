#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= AUTOCAST=1
bash tools/env/_py14.sh e320/t_real.py step_attach 2>&1 | grep -v "^_py14" | tail -n 9
