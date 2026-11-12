#!/bin/bash
# E302: the pair head with degree heads and a capacitated Sinkhorn plan, through the launcher with the card of experiments/E302_capacitated-pair-assignment.  usage: start_cap.sh <seed> <maps arm j0|j1|j2> [smoke]
cd "$(dirname "$0")/../.."
S=$1; MA=${2:-j1}
MAPS=$(pwd)/runs/e286_${MA}_s$S/model.pt
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E302.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e302_smoke_s$S --seed $S --maps $MAPS --capacity --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E302.k_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A degree head (out-degree of every entry, in-degree of every exit) and a capacitated Sinkhorn with a dustbin deliver pairs with a higher precision at the recall of the dual-softmax head, because an entry may feed several exits (27 % of the entries) and an exit take several entries (33 %)." \
  --pred "pair precision at the recall of E287 p2 +0.05 over p2 (the band is 0.03); topo F1 and pair recall within the guards" \
  --rule "E302 card section 0: by hand (jpair_pr.py score --control e287_p2_s<S>, jends_eval.py score for topo F1); KEEP when the precision at the control's recall rises by more than 0.03 on both seeds with topo F1 -0.01 and pair recall -0.02 held; TIE kept and refined (D82): the degree head with the SD lane counts; CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e302_k_s$S --seed $S --maps $MAPS --capacity --steps 3000 --workers 3
