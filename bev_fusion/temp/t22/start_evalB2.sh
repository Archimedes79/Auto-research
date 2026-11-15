#!/bin/bash
# scratch: evaluation of the stage-B2 run t22.B2_s0 (VAL 1000 frames, the flags of the gate A (ii) / stage B evaluations); diagnostic through the launcher (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
exec python3 ../autoresearch/tools/run.py --id t22.B2_s0.eval --gpu --gpu-max-mb 3000 --split val --no-card -- \
  bash -c "$PY -u evaluation/evaluate.py runs/t22_B2_s0 --quant --curves --fusion --velocity --n 1000 >> runs/t22_B2_s0_eval.log 2>&1"
