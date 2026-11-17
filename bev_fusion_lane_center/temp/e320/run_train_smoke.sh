#!/bin/bash
# E320 scratch: the dev jends_train main() end to end on the CPU: --smoke (20 steps), batch 2, --attach --geom-w 0.5 --e2e --scales
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
rm -rf runs/e320_smoke_train_cpu_s0 runs/e320_smoke_train_cpu_s0_maps
bash tools/env/_py14.sh e320/t_real.py train_smoke 2>&1 | tail -n 14
ls runs/e320_smoke_train_cpu_s0
