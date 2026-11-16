#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rm -f temp/e309/run_unsnap.done
for S in 0 1; do bash tools/env/_py14.sh evaluation/jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 > temp/e309/e312_unsnap_s$S.log 2>&1; done
echo done > temp/e309/run_unsnap.done
