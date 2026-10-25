#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# deployed-path rows for j1rcs1 on the idle machine, then the paired bootstrap with all systems
R=$BFLC
P=$R/tools/env/_py14.sh
for s in 0 1; do
  bash $P p16_pipeline_eval.py --name e18_D7_j1rcs1 --M 192 --K0 96 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
done
bash $P p18_final_boot.py > $R/runs/e18/final_boot.log 2> $R/runs/e18/final_boot.err
tail -4 $R/runs/e18/final_boot.log
