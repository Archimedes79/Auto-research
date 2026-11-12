#!/bin/bash
# E319 stage 1: the junction maps head with the heading-guided exchange (--dirx), the E307 recipe otherwise (3000 steps, heat sigma 1.0 m), through the launcher with the card of experiments/E319_heading-guided-exchange.  usage: start_maps.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E319.m_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The junction maps head has an isotropic context; a forward / backward exchange along the lane heading (ahead / behind at 2, 4, 8 m, two rounds, differentiable) gives the ends the evidence of the continuing lane: Mirko's hypothesis that the right shape needs a forward-backward information exchange." \
  --pred "topo F1 +0.005 over E312 of the same seed on the snapped read-out at p >= 0.1 (a TIE is the expected outcome); maps keypoint F1 +0.015, entry recall beyond 35 m +0.03" \
  --rule "E319 card section 0: by hand (jmaps_eval.py score for the maps stage; jends_eval.py score --snap --thr 0.1 for the read-out); KEEP when topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept (the head without the block is cheaper); CUT below -0.01 or the kill (stage 1 below E307 by 0.03); VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e319_m_s$S --seed $S --ctx dil --sigma 1.0 --steps 3000 --workers 3 --dirx
