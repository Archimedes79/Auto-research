#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
R=$BFLC
L="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"  # the folder of this launcher: its sibling scripts sit next to it
bash $L/_run_j3.sh a 0 &
bash $L/_run_j3.sh c 0 &
python3 $R/tools/experiment/run.py --id E232.eval --gpu-max-mb 9500 --retries 2 --basis E232.a0,E232.c0 --seed 0 \
  --hyp "evaluation of the long joint fine-tune arms on the failure series" -- bash $L/_eval_j3.sh > $R/runs/e17/logs/e232_eval.log 2>&1 &
wait
