#!/usr/bin/env bash
# one-off: CPU equivalence of the build on the hurdle model (object_moving): the stopped long run's checkpoint at step 5030,
# old code (HEAD snapshot in temp/arch_build/old) against the working tree, 3 val frames
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4 EQUIV_DEVICE=cpu EQUIV_RUN=runs/t10_dual_long_s0 EQUIV_CKPT=ckpt_00005030.pt EQUIV_REF=temp/arch_build/ref_hurdle.pt
P=~/miniconda3/envs/gcar/bin/python
grep -o '"object_moving": [a-z]*' runs/t10_dual_long_s0/config.json
EQUIV_CODE=temp/arch_build/old $P temp/arch_build/equiv.py save --n 3 2>&1 | tail -1
$P temp/arch_build/equiv.py compare 2>&1 | tail -2 | cut -c1-300
