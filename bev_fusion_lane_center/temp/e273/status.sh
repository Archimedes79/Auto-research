#!/bin/bash
# one-off: status of the E273 seed 1 run
cd "$(dirname "$0")/../.."
date
for p in 4 5 6 7; do
  f=runs/e273_p_s1_p$p/log.txt
  if [ -f "$f" ]; then echo "== p$p"; tail -n 3 "$f"; fi
done
ls runs | grep e273_p_s1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
