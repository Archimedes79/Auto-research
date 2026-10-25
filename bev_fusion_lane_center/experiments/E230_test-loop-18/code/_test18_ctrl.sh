#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# TEST (once): control E94 with the route head along the edges (graph metrics of the control on TEST: loop 16, S 0.702)
R=$BFLC
O=$R/runs/diag_test18
mkdir -p $O
export NMAXV=560
set -e
for SD in e104br e104br_s1; do
  bash $R/tools/env/_py14.sh p10_route_eval2.py --tok tok_e94_ar_test_clean --split test_clean --model $SD --kind x --taus 0.5 --cells exact,s1,s2,s3,wrong,none --tag test18_ctrl_$SD > $O/ctrl_$SD.log 2>&1
done
