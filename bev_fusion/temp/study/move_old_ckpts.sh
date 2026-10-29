#!/usr/bin/env bash
# One-off: C: nearly full -> the SUPERSEDED checkpoints of the t1 runs (all but the last, which evaluation uses) go to D: (copy, size check, then remove).
cd "$(dirname "$0")/../.." || exit 1
D=/mnt/d/Auto-research/bev_fusion_ckpts
for r in runs/t1_r0 runs/t1_r3_combined runs/t1_r4_soa; do
  a=$(basename $r); mkdir -p $D/$a
  for f in $(ls $r/ckpt_*.pt | head -n -1); do
    cp "$f" $D/$a/ && [ "$(stat -c %s "$f")" = "$(stat -c %s $D/$a/$(basename $f))" ] && rm "$f" && echo "moved $f"
  done
  echo "superseded checkpoints moved to $D/$a/ on $(date +%F) (C: full); the last one stays" > $r/MOVED.txt
done
