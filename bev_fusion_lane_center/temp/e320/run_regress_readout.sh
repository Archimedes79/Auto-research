#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e320/t_real.py regress_readout 2>&1 | grep -v "^_py14" | grep -E "regress_readout ok|Error|assert|Traceback" | head -5
