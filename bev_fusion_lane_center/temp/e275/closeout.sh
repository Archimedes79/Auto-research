#!/bin/bash
# E275 close-out, automatic part: waits for temp/e275/SCORED, then the CPU analysis of the dropped nodes and the verdict by the written rule; output temp/e275/closeout.log. The ledger notes, card section 7 and results
# README are written from this output by hand (the 00:17 / 22:43 check of the lane chat).
cd "$(dirname "$0")/../.."
until [ -f temp/e275/SCORED ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
{
  echo "== analyse $(date)"
  nice -n 12 bash tools/env/_py14.sh analyse.py e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1 2>&1 | tail -60
  echo "== verdict275 $(date)"
  nice -n 12 bash tools/env/_py14.sh qgraph_run_eval.py verdict275 --var e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --ctl e273_p_s0 e273_p_s1 2>&1 | tail -60
  echo "== baseline per tier, seed 1 $(date)"
  nice -n 12 bash tools/env/_py14.sh analyse_tiers.py e273_p_s1 --store /mnt/c/Auto-research/data/autogt/lane/0702a065c6 2>&1 | tail -14
  echo "== done $(date)"
} > temp/e275/closeout.log 2>&1
touch temp/e275/CLOSEOUT_DONE
