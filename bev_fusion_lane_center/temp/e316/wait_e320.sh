#!/bin/bash
# waits until both E320 pair runs have logged the 30 rows (steps 100 - 3000) that the kill statistic of the card needs, then prints the attlog of both seeds
cd "$(dirname "$0")/../.."
n() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
until [ "$(n runs/e320_a_s0/log.jsonl)" -ge 30 ] && [ "$(n runs/e320_a_s1/log.jsonl)" -ge 30 ]; do sleep 30; done
echo "E320 step 3000 logged $(date)"
for S in 0 1; do bash tools/env/_py14.sh jdiag.py attlog e320_a_s$S 2>&1 | tail -n 1; done
