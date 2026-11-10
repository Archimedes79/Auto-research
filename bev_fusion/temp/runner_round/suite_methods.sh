#!/bin/bash
# scratch: what each suite says about itself (read-only)
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
P=~/miniconda3/envs/gcar/bin/python
for s in trainsuite runsuite evalsuite; do
  echo "=== $s methods"
  $P -m $s methods 2>&1 | head -60
  echo "exit ${PIPESTATUS[0]}"
done
echo "=== suitecli head"
head -40 suitecli.py
