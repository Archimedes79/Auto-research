#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# numbers-review follow-ups on the idle machine: (1) loop-16b and loop-18 f1 timings repeated next to each other, j1 timed at the same pruning threshold as j1rc; (2) deployed-path rows of j1 with pruning 0.4 (the comparison base of j1rc)
R=$BFLC
P=$R/tools/env/_py14.sh
for k in a b; do
  bash $P p18_latency.py --name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --union 1 --fork_prune 0 --tag _f1_loop16b_re$k > /dev/null 2>&1
  bash $P p18_latency.py --name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --union 2 --fork_prune 0.3 --tag _f1_loop18_re$k > /dev/null 2>&1
done
bash $P p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _j1_fp04 > /dev/null 2>&1
bash $P p18_latency.py --name e18_D7_j1rc --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _j1rc_pairc > /dev/null 2>&1
bash $P p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _j1_fp04b > /dev/null 2>&1
for s in 0 1; do
  bash $P p16_pipeline_eval.py --name e123_D7_r3_j1 --M 192 --K0 96 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
done
for f in $R/runs/e18/latency/*re[ab]*.json $R/runs/e18/latency/*fp04*.json $R/runs/e18/latency/*pairc*.json; do echo $(basename $f) $(grep -A2 '"builder"' $f | grep median_ms | tr -d ' ,') $(grep -A2 '"total"' $f | grep median_ms | tr -d ' ,'); done
