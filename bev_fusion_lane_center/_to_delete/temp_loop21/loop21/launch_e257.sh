#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E257_dp-output-simplification/dp_sweep.txt
nohup python3 ../autoresearch/tools/run.py --id E257.a0 --retries 0 \
  --hyp "E257: Douglas-Peucker simplification (0.3 m) of the assembled graph removes at least a third of the vertices at equal Q" \
  --pred "vert100 a - c = -5.0 (tolerance 1.5) on fast_recall and champion; Q change -0.001 (tolerance 0.003)" \
  --rule "KEEP if vert100 falls by >= 3.0 on both models and the guards hold (Q loss <= 0.003, F1@0.5 <= 0.005, topo F1 <= 0.005, J <= 0.01); else TIE or CUT" \
  -- bash -c "bash tools/env/_py14.sh temp/loop21/e257_simplify.py > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
