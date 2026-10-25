#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# TEST (once) of the recall variant selected on VAL (rc1 + fork pruning 0.4: Q 0.740 is the best of rc1 / rc2); starts after the control run of the TEST chain
R=$BFLC
L=$R/temp/launchers/loop18
X=$R/experiment_execution/runs
until [ -f $X/E230.ctrl/done ]; do sleep 30; done
python3 $R/tools/experiment/run.py --split test --id E230.rc --gpu-max-mb 9000 --retries 1 --seed 0 --basis E219,E215 \
  --hyp "recall variant rc1 + fork pruning 0.4 on TEST" --pred "Q above the control, lanes found above 0.74, J above 0.38" --rule "report; nothing is selected on TEST" -- \
  bash $L/_pipeline18b.sh test_clean test18 e123_D7_r3_rc1 288 144 0.4 rcp
