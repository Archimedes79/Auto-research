#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E215b / E218: fork pruning (K190, 0.3) under sensor failure and in the best chain of loop 18 (map builder B + learned gate + registration + trust gate + fork pruning); failure series, VAL
R=$BFLC
RG=$R/runs/e18/regis_eval_a
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6 fork_prune=0.3"
set -e
# (1) champion without recursion, single frame and spatial hold, with / without pruning is e132_norec (no pruning) vs this run
bash $R/tools/env/_pyr.sh p17_eval.py --out e18_fp --perts dark,blur,cam_front3_off --variants single,maxdecay:0.95 --prev 0 --asm $ASM --suffix _fp03
# (2) best chain: B + gate series, exact / registered sigma 2 (thr 0.75) / none / wrong (gated 0.75), with pruning
E="bash $R/tools/env/_pyr.sh p17_eval.py --out e18_best --perts dark,blur,cam_front3_off --mapnet 1 --ckpt e17_map_b --variants gateB --asm $ASM"
$E --map_mode exact --map_sigma 0 --suffix _fp03_m0
$E --map_mode exact --map_sigma 2 --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _fp03_R75_s2
$E --map_mode none --suffix _fp03_mnone
$E --map_mode wrong --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _fp03_R75_wrong
