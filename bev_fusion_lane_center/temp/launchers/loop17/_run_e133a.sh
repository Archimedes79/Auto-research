#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E133a: failure-trained dense head (e65_ftall, loop 5) as the input stage of the champion builder: maps, series cache, evaluation
S=$REPO/scripts
L=$REPO/out/loop2/e17/logs
bash $S/_py14.sh p17_dense.py --head e65_ftall --tag ftall > $L/e133a_dense.log 2>&1 || exit 1
bash $S/_py14.sh p17_series.py --dense ftall --variants single,maxdecay:0.95 > $L/e133a_series.log 2>&1 || exit 1
bash $S/_py14.sh p17_eval.py --out e133a_ftall --variants single_ftall,maxdecay:0.95_ftall > $L/e133a.log 2>&1
