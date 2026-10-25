#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# back-to-back idle timing of j1 and j1rc (same architecture, different weights), alternating, to see whether the 21 ms of the first j1rc timing was machine state
R=$BFLC
P=$R/tools/env/_py14.sh
for k in a b; do
  bash $P p18_latency.py --name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.3 --tag _j1_pair$k > /dev/null 2>&1
  bash $P p18_latency.py --name e18_D7_j1rc --M 192 --K0 96 --rec 0 --union 2 --fork_prune 0.4 --tag _j1rc_pair$k > /dev/null 2>&1
done
for f in $R/runs/e18/latency/*pair*.json; do echo $f $(grep -A2 '"total"' $f | grep median_ms); done
