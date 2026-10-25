#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# FINAL VAL table and ms (run ONLY on an idle machine: no other GPU job, no training, no TEST chain):
#  (1) deployed path on all VAL frames (p16_pipeline_eval: CUDA-graph builder, assembly, route head along the edges, route success, ms per frame) for f1 / j1 / the recall variants with fork pruning, route heads s0 / s1
#  (2) stage times (p18_latency): assembly of loop 16b (FastUnion, no pruning) against the numba union + pruning
# usage: _run_final_val.sh <rc run name> <rc weight tag>   (e.g. e18_D7_rc2 rc2)
R=$BFLC
P=$R/tools/env/_py14.sh
rc=$1; rct=$2
set -e
for s in 0 1; do
  bash $P p16_pipeline_eval.py --name e123_D7_r3_f1 --M 288 --K0 144 --route p16_f2hr_route_s$s --fork_prune 0.3 --tag _final_s${s}_fp0.3
  bash $P p16_pipeline_eval.py --name e123_D7_r3_j1 --M 192 --K0 96 --route p16_f2hr_route_s$s --fork_prune 0.3 --tag _final_s${s}_fp0.3
  bash $P p16_pipeline_eval.py --name e123_D7_r3_rc1 --M 288 --K0 144 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
  [ -n "$rc" ] && bash $P p16_pipeline_eval.py --name $rc --M 288 --K0 144 --route p16_f2hr_route_s$s --fork_prune 0.4 --tag _final_s${s}_fp0.4
done
bash $P p18_latency.py --name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --union 1 --fork_prune 0 --tag _f1_loop16b
bash $P p18_latency.py --name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --union 2 --fork_prune 0.3 --tag _f1_loop18
bash $P p18_latency.py --name e123_D7_r3_f1 --M 288 --K0 144 --rec 1 --union 2 --fork_prune 0.3 --tag _f1rec_loop18
bash $P p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.3 --tag _j1_loop18
