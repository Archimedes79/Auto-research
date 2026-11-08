#!/usr/bin/env bash
# one-off: CPU equivalence of the final tree against the code before the build (temp/arch_build/old = HEAD snapshot): the base model
# (t10_dual_s0, step 3000, 4 frames) and the hurdle model (t10_dual_long_s0, step 5030, 3 frames); the references were saved with the old code
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4 EQUIV_DEVICE=cpu
P=~/miniconda3/envs/gcar/bin/python
EQUIV_REF=temp/arch_build/ref_cpu.pt $P tests/equivalence.py compare 2>&1 | tail -2
EQUIV_RUN=runs/t10_dual_long_s0 EQUIV_CKPT=ckpt_00005030.pt EQUIV_REF=temp/arch_build/ref_hurdle.pt $P tests/equivalence.py compare 2>&1 | tail -2
