#!/bin/bash
# scratch: one stage-B-type run of the twin / R-arm chain through trainsuite start (as start_B2.sh): $1 = run id, $2 = config name (training/configs/<name>.json), $3 = text key (R | S1 | C1), $4 = dry
# (dry = preview of the composed config and the static check only, no run). Without dry it starts the run detached (--go) and waits until the launcher of that id is gone
# (window_gate.py needs a command that lives as long as the job).
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
source ../bev_fusion/temp/t22/twinR_texts.sh
ID=$1; CFG=$2; K=$3
eval "HYP=\$HYP_$K; PRED=\$PRED_$K; RULE=\$RULE_$K"
[ -n "$HYP" ] && [ -n "$PRED" ] && [ -n "$RULE" ] || { echo "no texts for key $K"; exit 2; }
if [ "$4" = dry ]; then
  python3 -m trainsuite start ../bev_fusion/training/configs/$CFG.json --id $ID --hyp "$HYP" --pred "$PRED" --rule "$RULE" 2>&1 | cut -c1-400
  exit ${PIPESTATUS[0]}
fi
python3 -m trainsuite start ../bev_fusion/training/configs/$CFG.json --id $ID --hyp "$HYP" --pred "$PRED" --rule "$RULE" --go --wait 20 2>&1 | cut -c1-300
echo "$(date +%H:%M:%S) launched $ID; waiting for its launcher to end"
sleep 60
while ps -eo args | grep -q "[r]un.py --id $ID "; do sleep 30; done
echo "$(date +%H:%M:%S) launcher of $ID is gone"
