#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
rm -f runs/e320_smoke_cpu_s0/ends_card*.json
THR=0.5 bash tools/env/_py14.sh e320/t_real.py score_attach 2>&1 | grep -v "^_py14" | tail -n 3
ls runs/e320_smoke_cpu_s0 | grep ends_card
bash tools/env/_py14.sh e320/t_real.py kperr_att 2>&1 | grep -v "^_py14" | tail -n 5
