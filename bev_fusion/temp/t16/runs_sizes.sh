#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
for d in runs/*/; do
  n=$(basename $d)
  c=$(ls $d/ckpt_*.pt 2>/dev/null | wc -l)
  s=$(du -sm $d 2>/dev/null | cut -f1)
  printf "%6s MB  %2s ckpt  %s\n" "$s" "$c" "$n"
done | sort -rn
