#!/usr/bin/env bash
# one-off: the splat gate, the bf16 training clip and the clip equivalence after the dtype fix
cd "$(dirname "$0")/../.." || exit 1
P=~/miniconda3/envs/gcar/bin/python
OMP_NUM_THREADS=4 $P training/splat.py 2>&1 | tail -1
bash temp/arch_build/dbg_bf16.sh
for spec in "runs/t10_dual_s0 ckpt_00003000.pt base" "runs/t10_dual_long_s0 ckpt_00005030.pt hurdle"; do
  set -- $spec
  OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= EQUIV_DEVICE=cpu EQUIV_RUN=$1 EQUIV_CKPT=$2 EQUIV_REF=temp/arch_build/ref_clip_$3.pt \
    $P tests/equivalence.py compare 2>&1 | grep -v Warning | tail -1
done
