#!/usr/bin/env bash
# one-off: the clip-mode equivalence (2 keyframes, memory carried, one sample mirrored, state compared) of the bug-fixed tree against
# the code before the build (temp/arch_build/old), CPU, for the base model and the hurdle model
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=6 EQUIV_DEVICE=cpu CUDA_VISIBLE_DEVICES=
P=~/miniconda3/envs/gcar/bin/python
for spec in "runs/t10_dual_s0 ckpt_00003000.pt base" "runs/t10_dual_long_s0 ckpt_00005030.pt hurdle"; do
  set -- $spec
  export EQUIV_RUN=$1 EQUIV_CKPT=$2 EQUIV_REF=temp/arch_build/ref_clip_$3.pt
  echo "=== $3"
  EQUIV_CODE=temp/arch_build/old $P tests/equivalence.py save --n 3 2>&1 | grep -v Warning | tail -1
  $P tests/equivalence.py compare 2>&1 | grep -v Warning | tail -2 | cut -c1-260
done
