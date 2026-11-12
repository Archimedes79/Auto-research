#!/bin/bash
# E308: the pair head with the node tokens of B, through the launcher with the card of experiments/E308_pair-head-with-node-tokens.  usage: start_ntok.sh <seed> [maps arm j0|j1|j2] [smoke]
cd "$(dirname "$0")/../.."
S=$1; MA=${2:-j1}
MAPS=$(pwd)/runs/e286_${MA}_s$S/model.pt
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E308.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e308_smoke_s$S --seed $S --maps $MAPS --ntok --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E308.t_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The node tokens of B at the keypoints give the pair head what the E284 head had (AUC 0.974 among true ends) next to the maps' features (E287 p2: 0.930), so the pair decision on the better keypoints beats p2." \
  --pred "pair precision at the recall of E287 p2 +0.12 (AUC among true ends 0.97); pair recall at logit >= 0 about 0.25" \
  --rule "E308 card section 0: by hand (jpair_pr.py score --control e287_p2_s<S>, jends_eval.py score); KEEP when the precision at the control's recall rises by more than 0.03 on both seeds with topo F1 -0.01 and pair recall -0.02 held; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e308_t_s$S --seed $S --maps $MAPS --ntok --steps 3000 --workers 3
