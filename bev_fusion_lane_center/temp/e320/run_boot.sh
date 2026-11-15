#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e320/t_real.py readouts_new 2>&1 | grep -E "Traceback|Error" | head -3
ls runs/e320_smoke_cpu_s0 | grep counts
bash tools/env/_py14.sh e320/t_real.py boot_test 2>&1 | grep -v "^_py14" | tail -n 8
