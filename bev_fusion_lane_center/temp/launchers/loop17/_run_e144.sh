#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
# E144 (E212 in the report): the planner-side consumer on the graphs of the failure-half frames: champion chains and the map builder B, and the same frames without perturbation (matched reference)
S=$BFLC/training
L=$BFLC/runs/e17/logs
SER=""
for p in dark blur cam_front3_off; do SER="$SER,${p}_single,${p}_maxdecay_0.95,${p}_gate"; done
SERB=""
for p in dark blur cam_front3_off; do SERB="$SERB,${p}_single,${p}_maxdecay_0.95,${p}_gateB"; done
bash $BFLC/tools/env/_py14.sh p17_consumer.py --out e144_champ --series ${SER#,},dark_single_cleanref,dark_maxdecay_0.95_cleanref,dark_gate_cleanref > $L/e144_champ.log 2>&1
bash $BFLC/tools/env/_py14.sh p17_consumer.py --out e144_map --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 0 --series ${SERB#,},dark_single_cleanref,dark_maxdecay_0.95_cleanref,dark_gateB_cleanref > $L/e144_map.log 2>&1
echo finished > $L/e144.done
