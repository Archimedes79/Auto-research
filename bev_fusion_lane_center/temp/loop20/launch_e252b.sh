#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E252_gt-map-offset-by-city/gt_offset_controls.txt
nohup python3 ../autoresearch/tools/run.py --id E252.b0 --retries 0 \
  --hyp "E252 after the referee: the Boston - Singapore gap of Q is outside a permutation null of city labels, and the synthetic shift of the ground truth gives the metres scale; per-polyline offsets are small" \
  --pred "permutation p of the observed gap > 0.05 (22 scenes only); a 0.5 m shift costs Q about 0.05; per-polyline median offset above 0.3 m in under 10 % of the polylines" \
  --rule "report only; the E252 verdict (cap not supported) stands unless the Boston gap is NEGATIVE, outside the null and the per-polyline share above 0.3 m is at least 1.5 x Singapore" \
  -- bash -c "bash tools/env/_py14.sh e252b_controls.py champion > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
