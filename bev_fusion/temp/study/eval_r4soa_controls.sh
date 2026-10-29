#!/usr/bin/env bash
# One-off (t3 try 1): the sanity controls on the long-trained t1_r4_soa (7000 steps x accum 2), val 82 scenes, 1000 frames as the t2 arms
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
for c in "" shuffle_radar zero_cam shuffle_cam; do
  echo "== r4soa control '$c' $(date +%T)"
  OMP_NUM_THREADS=2 nice -n 5 $PY -u evaluation/evaluate_run.py runs/t3_eval_r4soa --n 1000 --workers 2 --batch 4 --ckpt-dir runs/t1_r4_soa ${c:+--control $c} 2>&1 | tail -n 1
done
echo "r4soa controls finished $(date +%T)"
