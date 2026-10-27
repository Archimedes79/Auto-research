#!/usr/bin/env bash
# One-off (t2_block_swaps): train the arms one after the other from a pinned commit, then evaluate each on 1000 VAL keyframes.
# Usage: bash temp/study/lane.sh <commit> <arm> [<arm> ...]   -> runs/t2_<arm>/ (train.log, eval/), temp/study/lane.log
cd "$(dirname "$0")/../.." || exit 1
rev=$1; shift
for a in "$@"; do
  r=runs/t2_$a
  if [ -f $r/eval/metrics.json ]; then echo "$a already evaluated, skip"; continue; fi
  echo "== $a train $(date +%T)"
  [ -f $r/.done ] || bash temp/train/run_cfg.sh training/configs/t2_$a.json $rev
  [ -f $r/.done ] || { echo "$a: no .done, skipped evaluation"; continue; }
  echo "== $a eval $(date +%T)"
  ~/miniconda3/envs/gcar/bin/python -u evaluation/evaluate_run.py $r --n 1000 --workers 4 --batch 4 > temp/study/eval_$a.log 2>&1
  tail -n 1 temp/study/eval_$a.log
done
echo "lane finished $(date +%T)"
