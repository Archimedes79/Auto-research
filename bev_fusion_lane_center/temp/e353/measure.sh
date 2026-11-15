#!/bin/bash
cd "$(dirname "$0")/../.."
N="nice -n 19 bash tools/env/_py14.sh"
for S in 0 1; do $N jprune_b.py check --seed $S --max 24 --threads 2 > temp/e353/check24_s$S.log 2>&1; done
for S in 0 1; do $N e352_sec.py $S > temp/e353/sections_s$S.log 2>&1; done
for i in 1 2; do for T in 1 2; do $N jprune_b.py check --seed 0 --max 12 --threads $T 2>/dev/null | grep -E "seconds_per_frame" | sed "s/^/threads $T: /" >> temp/e353/threads_timing.log; done; done
echo "measure done $(date)" > temp/e353/measure.done
