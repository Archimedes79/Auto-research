#!/bin/bash
# scratch: gate A (ii) of the t22 card: the A2 student frozen in the one-frame chain t21.pair4_s0 (stand-in), then the live-code twin without a student; diagnostics through the launcher (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
PY=$HOME/miniconda3/envs/gcar/bin/python
for k in standin twin; do
  if [ ! -d runs/t22_${k}_pair4 ]; then
    if [ $k = standin ]; then $PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 runs/t22_standin_pair4 --student runs/t22_A2_s0 || exit 1
    else $PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 runs/t22_twin_pair4 --twin || exit 1; fi
  fi
done
for k in standin twin; do
  python3 ../autoresearch/tools/run.py --id t22.${k}_pair4 --gpu --gpu-max-mb 3000 --split val --no-card -- \
    bash -c "$PY -u evaluation/evaluate.py runs/t22_${k}_pair4 --quant --curves --fusion --velocity --n 1000 --ckpt-dir runs/t21_pair4_s0 >> runs/t22_${k}_pair4.log 2>&1" || echo "run $k failed: $?"
done
