#!/usr/bin/env bash
# one-off: the hurdle comparison again with 6 threads (the thread count the reference was saved with), raw log
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=6 EQUIV_DEVICE=cpu CUDA_VISIBLE_DEVICES=
L=temp/arch_build/equiv_hurdle6_raw.log
EQUIV_RUN=runs/t10_dual_long_s0 EQUIV_CKPT=ckpt_00005030.pt EQUIV_REF=temp/arch_build/ref_clip_hurdle.pt ~/miniconda3/envs/gcar/bin/python tests/equivalence.py compare > $L 2>&1
echo "exit code $?" >> $L
echo "=== done" >> $L
