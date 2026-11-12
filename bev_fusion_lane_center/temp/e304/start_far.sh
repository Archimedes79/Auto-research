#!/bin/bash
# E304: the junction maps head with a far-range weighted heat loss, through the launcher with the card of experiments/E304_far-range-weighted-maps.  usage: start_far.sh <seed> <ctx none|dil|attn> [smoke]
cd "$(dirname "$0")/../.."
S=$1; CTX=$2
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E304.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jmaps_train.py --name e304_smoke_s$S --seed $S --ctx $CTX --far-w 2 --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E304.f_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Counting the heat BCE of the cells beyond 35 m three times (far-w 2) raises the recall of the junction ends beyond 35 m (0.55 - 0.58 for E286 j1, 53 % of the GT ends) without hurting the near range." \
  --pred "far recall +0.04 over E286 j1 of the same seed; kp_f1 and near recall within the guards" \
  --rule "E304 card section 0: by hand (jmaps_eval.py range split); KEEP when the far recall rises by more than 0.03 on both seeds with kp_f1 within 0.01 and near recall within 0.02; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e304_f_s$S --seed $S --ctx $CTX --far-w 2 --steps 3000 --workers 3
