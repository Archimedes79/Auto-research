#!/bin/bash
# E313: the junction maps head with end heat sigma 0.7 m, through the launcher with the card of experiments/E313_heat-sigma-0.7.  usage: start_sigma.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E313.s_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A heat target of sigma 0.7 m separates and sharpens the junction ends further than sigma 1.0 m (E307: keypoint F1 +0.06 .. +0.09 over 1.5 m)." \
  --pred "keypoint F1 +0.02 over E307 of the same seed; crowded entry recall not lower" \
  --rule "E313 card section 0: by hand (jmaps_eval.py score, maps_cmp.py); KEEP when the keypoint F1 rises by more than 0.01 on both seeds with the crowded entry recall -0.03 and the zone IoU within 0.01; TIE inside the band (E307 stays); CUT below -0.01; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e313_s_s$S --seed $S --ctx dil --sigma 0.7 --steps 3000 --workers 3
