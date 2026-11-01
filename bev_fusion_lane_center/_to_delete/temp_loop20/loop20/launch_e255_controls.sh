#!/bin/bash
# E255 matched-vertex procedure for the two exact-map controls (CPU, 2 processes), beside the pair-3 trainings
cd "$(dirname "$0")/../.."
OUT=experiments/E255_poisson-set-loss-matched-vertex/matched_controls.txt
nohup python3 ../autoresearch/tools/run.py --id E255.mc --retries 0 \
  --hyp "E255 controls: the plain j1eq fine-tunes on the exact map evaluated at the matched vertex budget (14.3 per 100 m) by the declared procedure" \
  --pred "the chosen points keep Q within 0.01 of the stored cards (0.731 / 0.732 at tau 0.7)" \
  --rule "procedure of the E255 card; a control whose chosen point is outside [14.15, 14.45] is reported as nearest" \
  -- bash -c "bash tools/env/_py14.sh e255_matched.py e24_j1ex_s0 e24_j1ex_s1 > $OUT 2>&1" > /dev/null 2>&1 &
sleep 2; echo launched
