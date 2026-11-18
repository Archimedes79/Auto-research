#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
R=$BFLC
bash $R/tools/env/_py14.sh $R/experiments/E219_recall-weight-finetune-under-pruning/code/p18_final_boot.py > $R/runs/e18/final_boot.log 2> $R/runs/e18/final_boot.err
cat $R/runs/e18/final_boot.log
