#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E256_candidates-by-city/by_city.txt
nohup python3 ../autoresearch/tools/run.py --id E256.a0 --retries 0 \
  --hyp "E256: the ranking of the D4 deployment candidates by Q is the same in Boston and in Singapore One-North" \
  --pred "no pair flips by more than the band 0.010 between the cities" \
  --rule "a pair ordered one way in Boston and the other way in Singapore by more than 0.010 in both cities is a flip and goes into D4; otherwise the ranking is confirmed" \
  -- bash -c "bash tools/env/_py14.sh e252_gt_offset.py fast_recall recall fast control champion > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
