#!/bin/bash
# E318 scratch: the dump of a CPU-built unc checkpoint (8 VAL frames), then a look at the pickle
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e318/make_smoke.py 2>&1 | tail -n 2
bash tools/env/_py14.sh jends_eval.py dump --run e318_smoke_cpu_s0 --max 8 2>&1 | tail -n 4
bash tools/env/_py14.sh e318/look_dump.py 2>&1 | tail -n 4
