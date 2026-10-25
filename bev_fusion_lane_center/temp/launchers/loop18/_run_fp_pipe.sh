#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E215 (fork pruning K190): the deployed path (CUDA-graph builder, fast assembly, route head along the edges, seed 0 and 1) on all validation frames, with and without the un-linking of forks without junction-head support
R=$BFLC
for s in 0 1; do
  for fp in 0 0.3; do
    bash $R/tools/env/_py14.sh p16_pipeline_eval.py --name e123_D7_r3_f1 --M 288 --K0 144 --route p16_f2hr_route_s$s --fork_prune $fp --tag _s${s}_fp$fp
  done
done
