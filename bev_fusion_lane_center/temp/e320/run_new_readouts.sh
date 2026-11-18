#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e320/t_real.py readouts_new 2>&1 | grep -v "^_py14" | grep -E "^\[|^ends_card|^plain|^p0|Error|Traceback|assert" | cut -c1-330
