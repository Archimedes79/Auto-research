#!/bin/bash
# CPU smoke of the curve metric through the real adapter (6 frames of the shadow run): files and keys
cd "$(dirname "$0")/../.." || exit 1
R=runs/t10_dual_s0_gen
mkdir -p $R
cp runs/t10_dual_s0/config.json runs/t10_dual_s0/gt.json $R/
ln -sfn ../t10_dual_s0/ckpt_00003000.pt $R/ckpt_00003000.pt
ln -sfn ../t10_dual_s0/code $R/code
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py $R --device cpu --curves --workers 1 eval --n 6 > temp/pitch/curves_smoke.log 2>&1
echo "exit $?"
grep -E 'curve/' $R/eval/table.txt | head -14
tail -3 temp/pitch/curves_smoke.log | cut -c1-200
