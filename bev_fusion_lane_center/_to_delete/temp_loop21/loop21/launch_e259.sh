#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E259_assembly-cleanup-ablation/clean_ablation.txt
nohup python3 ../autoresearch/tools/run.py --id E259.a0 --retries 0 \
  --hyp "E259: the spur / bubble clean-up of the assembly adds nothing on top of fork pruning and can be removed (simpler, faster)" \
  --pred "Q change -0.001 (tol 0.003); saving 0.9 ms relative" \
  --rule "drop only if on all seven models Q <= -0.003, J <= -0.01, F1@0.5 <= -0.005, vert100 <= 0.3, FL <= 0.01 with and without dp_eps 0.2 and saving >= 0.5 ms" \
  -- bash -c "bash tools/env/_py14.sh temp/loop21/e259_clean.py > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
