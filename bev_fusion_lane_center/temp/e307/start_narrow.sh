#!/bin/bash
# E307: the junction maps head with narrower end heat targets, through the launcher with the card of experiments/E307_narrow-end-heat.  usage: start_narrow.sh <seed> [smoke]
cd "$(dirname "$0")/../.."
S=$1
if [ "$2" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E307.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jmaps_train.py --name e307_smoke_s$S --seed $S --ctx dil --sigma 1.0 --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E307.n_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Entry / exit heat targets of sigma 1.0 m instead of 1.5 m separate the ends of neighbouring lanes (3.5 m apart), which the 3 x 3 peak read-out merges today (crowded entry recall 0.54 against 0.66 isolated)." \
  --pred "kp F1 +0.02 and crowded entry recall +0.06 over E286 j1 of the same seed; isolated ends within 0.01" \
  --rule "E307 card section 0: by hand (jmaps_eval.py score); KEEP when kp F1 rises by more than 0.01 and the crowded recall by more than 0.04 on both seeds with zone IoU within 0.01; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e307_n_s$S --seed $S --ctx dil --sigma 1.0 --steps 3000 --workers 3
