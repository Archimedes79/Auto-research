#!/bin/bash
# E312 maps arm m2: heat sigma 1.0 m + far-range weight 2 on the dilated-context maps head, through the launcher with the card of experiments/E312_combined-end-maps.  usage: start_maps.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E312.m2_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Narrow end heat (sigma 1.0 m: crowded ends) together with a far-range weighted heat loss (ends beyond 35 m) finds more junction ends than the maps of E286 j1, so that more pairs have both ends found (52 % today, strict match), and the E305 pair head on them lifts the snapped read-out." \
  --pred "topo F1 +0.01 over E305 of the same seed on the snapped read-out at p >= 0.1; keypoint F1 of the maps within 0.03 of E286 j1 or better" \
  --rule "E312 card section 0: by hand (jmaps_eval.py score for the maps; jends_eval.py score --snap --thr 0.1 for the read-out); KEEP when topo F1 over E305 rises by more than 0.01 on both seeds with the guards held; TIE kept and refined (D82); CUT below or the kill; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e312_m2_s$S --seed $S --ctx dil --sigma 1.0 --far-w 2 --steps 3000 --workers 3
