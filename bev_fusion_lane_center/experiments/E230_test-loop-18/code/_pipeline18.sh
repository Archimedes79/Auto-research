#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# Graph metrics + navigation attribute (route head along the edges, seeds 0 / 1) of a builder with fork pruning, on one split (val_clean for the dry run, test_clean for the one TEST use)
# usage: _pipeline18.sh <split> <tag> <builder run name> <M> <K0> <fork_prune> <short id>
R=$BFLC
split=$1; tag=$2; name=$3; M=$4; K0=$5; fp=$6; id=$7
O=$R/runs/diag_$tag
mkdir -p $O
ASM="cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6 fork_prune=$fp"
export NMAXV=560
set -e
bash $R/tools/env/_py14.sh p16_final_eval.py --design D7 --name $name --split $split --set M=$M K0=$K0 --asm $ASM --tau 0.7 --tag final18_$id --keep > $O/${id}_graph.log 2>&1
bash $R/tools/env/_py14.sh p16_tokens.py --design D7 --name $name --out tok_p18_$id --split $split --tau 0.7 --set M=$M K0=$K0 --asm $ASM > $O/${id}_tokens.log 2>&1
bash $R/tools/env/_py14.sh p16_tok_resample.py tok_p18_${id}_$split tok_p18_${id}r_$split 4.0 > $O/${id}_resample.log 2>&1
for SD in p16_f2hr_route_s0 p16_f2hr_route_s1; do
  bash $R/tools/env/_py14.sh p10_route_eval2.py --tok tok_p18_${id}r_$split --split $split --model $SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag ${tag}_$id > $O/${id}_$SD.log 2>&1
done
echo "done $id" >> $O/done.txt
