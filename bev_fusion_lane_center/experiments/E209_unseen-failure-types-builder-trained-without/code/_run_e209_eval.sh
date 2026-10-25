#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E209 evaluation only (the chain _run_e209.sh stopped after the second training: run.py of that moment crashed in ledger_write when the ledger columns were changed by another chat)
R=$BFLC
L=$R/runs/e17/logs
python3 $R/tools/experiment/run.py --id E209.eval --gpu-max-mb 9500 --retries 2 --basis E209.b2,E209.c2 --seed 0 \
  --hyp "evaluation of E209.b2 / c2 on the held-out night and blur series" -- bash $R/experiments/E209_unseen-failure-types-builder-trained-without/code/_eval_e209.sh > $L/e209_eval.log 2>&1 && echo finished > $L/e209.done
