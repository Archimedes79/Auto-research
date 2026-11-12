#!/bin/bash
# E306: the junction maps head on exact polygon targets, through the launcher with the card of experiments/E306_exact-polygon-targets.  usage: start_exact.sh <seed> [smoke]
cd "$(dirname "$0")/../.."
S=$1
if [ "$2" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E306.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jmaps_train.py --name e306_smoke_s$S --seed $S --ctx dil --raster exact --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E306.x_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Training and scoring the junction maps head on exact polygon targets (area rule) instead of the half-cell shifted legacy fill raises the zone IoU against the same exact exam." \
  --pred "zone IoU +0.03 over E286 j1 of the same seed scored on the exact targets; core IoU and keypoint F1 within the guards" \
  --rule "E306 card section 0: by hand (jmaps_eval.py score); KEEP when zone IoU rises by more than 0.02 on both seeds with core IoU -0.02 and kp_f1 -0.01 held; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e306_x_s$S --seed $S --ctx dil --raster exact --steps 3000 --workers 3
