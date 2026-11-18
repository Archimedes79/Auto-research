#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# TEST re-run of the j1 finalist (E230.j1 failed in its resample step: a concurrent edit of p16_loss.py had a missing import for one minute; settings unchanged), after the recall-variant run
R=$BFLC
L="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"  # the folder of this launcher: its sibling scripts sit next to it
X=$R/experiment_execution/runs
until [ -f $X/E230.rc/done ]; do sleep 30; done
python3 $R/tools/experiment/run.py --split test --id E230.j1p2 --gpu-max-mb 9000 --retries 0 --seed 0 --basis E215 \
  --hyp "single-frame j1 192 / 96 + fork pruning as the fast deployment setting (re-run of E230.j1p)" --pred "S >= 0.74, J >= 0.35" --rule "report; nothing is selected on TEST" -- \
  bash $L/_pipeline18b.sh test_clean test18 e123_D7_r3_j1 192 96 0.3 j1p
