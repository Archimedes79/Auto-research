#!/bin/bash
# fork C scratch: after the stored-dump verification, make the dumps again on the CPU (seed 0) and compare dumps and cards with the stored ones
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
until [ -f temp/e321/verify.done ]; do sleep 20; done
rm -f temp/e321/fresh.done
/usr/bin/time -v nice -n 12 bash tools/env/_py14.sh evaluation/jtest_eval.py verify --seed 0 --fresh --device cpu --boot 0 > temp/e321/fresh_s0.log 2>&1
echo done > temp/e321/fresh.done
