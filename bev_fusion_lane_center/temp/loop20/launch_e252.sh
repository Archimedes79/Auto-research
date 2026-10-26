#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=experiments/E252_gt-map-offset-by-city/gt_offset.txt
nohup python3 ../autoresearch/tools/run.py --id E252.a0 --retries 0 \
  --hyp "E252: the map offset of the lane ground truth lowers Q by city (Boston worst) and shows as a per-scene lateral bias of correct predictions" \
  --pred "Q(Boston) - Q(Singapore) = -0.03 for champion, j1eq, j1rc; share |offset| > 0.5 m at least 1.5 x higher in Boston" \
  --rule "CAP if all three Q gaps <= -0.03 with non-overlapping 95 % scene intervals and the |offset| > 0.5 m share ratio >= 1.5; NO EVIDENCE if the gap is within +-0.010; else report by city only" \
  -- bash -c "bash tools/env/_py14.sh e252_gt_offset.py > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
