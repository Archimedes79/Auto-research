#!/bin/bash
# scratch: the follow-up chain of the baseline: extraction job t17.feat, run t17.data, its evaluation, (the control is relaunched behind them by hand)
cd "$(dirname "$0")/../.." || exit 1               # bev_fusion/
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
RUN=/mnt/c/Auto-research/autoresearch/tools/run.py
nohup python3 $RUN --id t17.feat --gpu --split train --no-card --after t16.full_s0.eval -- bash temp/t16/prep_more.sh extract > runs/t17_feat.launcher.log 2>&1 &
sleep 3
cd ../autoresearch
HYP="Adding the 272 train scenes of nuScenes parts 7 - 10 (700 train scenes instead of 428; the 82 val scenes unchanged) to the full network's recipe reduces the train-to-val gap and improves the maps and objects on the same 1000 val frames; nothing else changes (t17 card)."
PRED="Against t16.full_s0 (single seed each, VAL 1000, GT v8, guesses): the six map IoUs +0.01 ... +0.04, front mAP2 +0.02, vehicle AP@2 +0.01; the generalisation gap smaller by at least 20 percent; vehicle moving EPE within +-0.3 m/s; ms per frame and parameters identical."
RULE="Single seed each, so by hand: ADOPT if at least four of the six map IoUs and front mAP2 beat t16.full_s0 by more than their band AND the paired bootstrap of the maps excludes 0 AND no guard is worse than its band (vehicle moving EPE +0.3, static EPE +0.1, front mAP2 -0.010); inside the bands: keep t16.full_s0; worse on two or more guards: keep it and record the negative (t17 card section 1)."
python3 -m trainsuite start ../bev_fusion/training/configs/t17_data.json --id t17.data --hyp "$HYP" --pred "$PRED" --rule "$RULE" --go --run-arg=--after --run-arg=t17.feat 2>&1 | grep -E 'check:|started|refus|error' | cut -c1-200
cd ../bev_fusion
sleep 3
nohup python3 $RUN --id t17.data.eval --gpu --split val --no-card --after t17.data -- bash temp/t16/eval_chain.sh runs/t17_data > runs/t17_data.eval.launcher.log 2>&1 &
sleep 3
ps -eo pid,etimes,cmd | grep 'run.py --id t1[67]' | grep -v grep | cut -c1-120
