#!/bin/bash
# train-vs-validation evaluation of t10_dual_s0 with the curve numbers, in a SHADOW run folder so the stored 1000-frame val evaluation of the run is not replaced:
# runs/t10_dual_s0_gen holds a copy of config.json / gt.json and links to the checkpoint and the code snapshot. 600 frames spread over the scenes of each split, cold memory.
cd "$(dirname "$0")/../.." || exit 1
R=runs/t10_dual_s0_gen
mkdir -p $R
cp runs/t10_dual_s0/config.json runs/t10_dual_s0/gt.json $R/
ln -sfn ../t10_dual_s0/ckpt_00003000.pt $R/ckpt_00003000.pt
ln -sfn ../t10_dual_s0/code $R/code
export OMP_NUM_THREADS=3
~/miniconda3/envs/gcar/bin/python -u evaluation/bev_suite.py $R --curves --workers 3 generalisation --n 600
