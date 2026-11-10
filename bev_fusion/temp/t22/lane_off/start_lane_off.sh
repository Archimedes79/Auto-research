#!/bin/bash
# scratch: the L4 check, two diagnostic evaluations of the checkpoint of runs/t20_base_s0 under the live code, one after the other through the launcher (no card, never feeds a verdict):
#   diag.t20_L4on   the twin: the lane branch on (it also tells whether the live evaluator reproduces the stored t20.base_s0 numbers)
#   diag.t20_L4off  the lane branch dropped after the load (temp/t22/lane_off/suite_lane_off.py)
# VAL 1000 frames, the flags of the other evaluations (--curves --fusion --velocity). About 10 minutes each on the GPU.
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
python3 ../autoresearch/tools/run.py --id diag.t20_L4on --gpu --gpu-max-mb 3000 --split val --no-card --retries 0 -- \
  bash -c "$PY -u evaluation/evaluate.py runs/diag_t20_L4on --quant --curves --fusion --velocity --n 1000 --ckpt-dir runs/t20_base_s0 >> runs/diag_t20_L4on.log 2>&1"
echo "twin exit $?"
python3 ../autoresearch/tools/run.py --id diag.t20_L4off --gpu --gpu-max-mb 3000 --split val --no-card --retries 0 -- \
  bash -c "$PY -u temp/t22/lane_off/suite_lane_off.py runs/diag_t20_L4off --device cuda --workers 2 --split val --ckpt-dir runs/t20_base_s0 --curves --fusion --velocity eval --n 1000 >> runs/diag_t20_L4off.log 2>&1"
echo "L4 off exit $?"
