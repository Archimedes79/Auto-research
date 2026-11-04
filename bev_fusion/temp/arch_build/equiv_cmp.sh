#!/usr/bin/env bash
# one-off: forward-only CPU equivalence of the live tree against the references saved with the code before the five-layer build
# (ref_clip_base.pt: t10_dual_s0 step 3000; ref_clip_hurdle.pt: t10_dual_long_s0 step 5030), 2 threads, raw log with the exit codes
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 EQUIV_DEVICE=cpu CUDA_VISIBLE_DEVICES=
P=~/miniconda3/envs/gcar/bin/python
L=temp/arch_build/equiv_cmp_raw.log
: > $L
for spec in "runs/t10_dual_s0 ckpt_00003000.pt base" "runs/t10_dual_long_s0 ckpt_00005030.pt hurdle"; do
  set -- $spec
  echo "=== $3" >> $L
  EQUIV_RUN=$1 EQUIV_CKPT=$2 EQUIV_REF=temp/arch_build/ref_clip_$3.pt $P tests/equivalence.py compare >> $L 2>&1
  echo "exit code $?" >> $L
done
echo "=== done" >> $L
