#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E231.a: the j1 family (j1, equal-budget control j1eq / j1eqs1, recall variants j1rc / j1rcs1) on the loop-17 failure series (night, blur, three cameras off; single frame and spatial hold), fork pruning 0.4, no time recursion; VAL, 22 scenes
R=$BFLC
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6 fork_prune=0.4"
set -e
for pair in j1:e123_D7_r3_j1 j1eq:e18_D7_j1eq j1rc:e18_D7_j1rc j1rcs1:e18_D7_j1rcs1 j1eqs1:e18_D7_j1eqs1; do
  tag=${pair%%:*}; name=${pair#*:}
  [ -f $R/runs/e123/$name/model.pt ] || { echo "skip $name (no model)"; continue; }
  bash $R/tools/env/_pyr.sh p17_eval.py --name $name --set M=192 K0=96 --out e18_j1fam --perts dark,blur,cam_front3_off --variants single,maxdecay:0.95 --prev 0 --asm $ASM --suffix _$tag
done
