#!/bin/bash
cd "$(dirname "$0")/../.."
until grep -q "run e324s\|run e331" temp/day14/day14.log 2>/dev/null; do sleep 20; done
sleep 120
echo "picked: $(grep -E '^run ' temp/day14/day14.log | tail -n 1)"
ps aux | grep -E "[j]ends_train|[r]un.py" | cut -c1-150
tail -n 2 temp/day14/e324_lane0.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
