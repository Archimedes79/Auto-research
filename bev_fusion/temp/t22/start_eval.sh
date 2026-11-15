#!/bin/bash
# scratch: evaluation of a stage-B-type run (VAL 1000 frames, the flags of the gate A (ii) / stage B / B2 evaluations); diagnostic through the launcher (no card, never feeds a verdict)
# $1 = run folder name (e.g. t22_R_s0), $2 = ledger id of the evaluation (e.g. t22.R_s0.eval)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
exec python3 ../autoresearch/tools/run.py --id $2 --gpu --gpu-max-mb 3000 --split val --no-card -- \
  bash -c "$PY -u evaluation/evaluate.py runs/$1 --quant --curves --fusion --velocity --n 1000 >> runs/$1_eval.log 2>&1"
