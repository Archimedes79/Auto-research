#!/bin/bash
# E315 stage 1: the junction maps head, 6000 steps, through the launcher with the card of experiments/E315_longer-maps-training.  usage: start_maps.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E315.m_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The junction maps and the E312 pair head are still improving at step 3000 (loss falling): twice the maps steps lift the maps and through them the snapped read-out." \
  --pred "topo F1 +0.01 over E312 of the same seed on the snapped read-out at p >= 0.1; maps keypoint F1 +0.02" \
  --rule "E315 card section 0: by hand (jmaps_eval.py score for the maps stage; jends_eval.py score --snap --thr 0.1 for the read-out); KEEP when topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept (the shorter recipe is cheaper); CUT below -0.01 or the kill (stage 1 below E307 by 0.03); VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e315_m_s$S --seed $S --ctx dil --sigma 1.0 --steps 6000 --workers 3
