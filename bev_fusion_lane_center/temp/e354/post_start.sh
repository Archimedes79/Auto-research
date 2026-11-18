#!/bin/bash
# E354: queue the post-processing of one seed behind its two trainings (run.py --after). usage: post_start.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E354.post_s$S --seed $S --retries 0 --require-card --split val --after E354.r_s$S,E354.z_s$S --wait-after 600 --timeout 300 \
  --hyp "E354: the read-outs of the recursive feature state (stream, off, shuffle, trunc), the control arm and the layer-1 control of the new stream-order driver on E341, scored unsnapped and snapped at p >= 0.1, with the flip rates at matched delivery." \
  --pred "the driver reproduces the stored E341 cards within 0.0005; r minus z +0.004 unsnapped topo F1; flip rate -3 points at matched delivery" \
  --rule "E354 card section 0 (by hand: replay); the layer-1 control must hold before any arm is read; VAL 873, no TEST" \
  -- bash temp/e354/post_seed.sh $S
