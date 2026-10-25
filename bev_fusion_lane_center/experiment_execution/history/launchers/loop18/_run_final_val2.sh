#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# deployed-path VAL rows (route success along the edges, ms per frame) for the two E219 additions, on an idle machine: j1rc (fast recall variant) and rc1s1 (seed-1 replicate of rc1), fork pruning 0.4, route heads s0 / s1; then the stage times of j1rc
R=$BFLC
P=$R/tools/env/_py14.sh
set -e
for s in 0 1; do
  bash $P p16_pipeline_eval.py --name e18_D7_j1rc --M 192 --K0 96 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
  bash $P p16_pipeline_eval.py --name e18_D7_rc1s1 --M 288 --K0 144 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
done
bash $P p18_latency.py --name e18_D7_j1rc --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _j1rc_loop18
