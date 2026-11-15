#!/bin/bash
# E352 STEP 0 (proposal, NOT started): 9 point sets (null, t1, t2, r1d0-2, r2d0-2) x 2 seeds: one B' + junction dump job per point (CPU, about 10-15 min each; 1 GPU slot would take about 2 min: add --gpu), then the scoring jobs.
# usage: bash temp/e352/step0_runs.sh <seed> <point> [dump|score]      (the lead decides the lanes / --after chain; `ls temp/` first)
# The null is the equality gate of the build: dump + score it FIRST (its `numbers.base` must be the stored B plain graph within topo F1 2e-4; its junction cards the stored E324 s cards).
cd "$(dirname "$0")/../.."
R=$PWD
S=$1; P=$2; WHAT=${3:-dump}
HYP="Narrowing B's pyramid trunk by the importance ranking of E345 (T1 / T2) costs B's own graph and the junction system on B' less than removing the same number of random channels: step 0 of E352, hook edits (channels replaced by their calibration means), no training."
PRED="T1: B plain graph -0.004 topo F1, junction snapped -0.004 (E352 card section 1); random control worse by at least 0.003; the null reproduces the stored B plain graph (0.6259 / 0.6249) within 2e-4"
RULE="E352 card section 0 kill (by hand: replay of the cards, evaluation/jprune_b.py): stop when T1 loses more than 0.03 topo F1 on B's plain graph without retraining and the random control loses no more; the null must reproduce the stored cards within 2e-4; VAL 873, no TEST"
case $WHAT in
  dump)
    python3 ../autoresearch/tools/run.py --id E352.s0_${P}_s$S --seed $S --ram-mb 4000 --retries 0 --require-card --split val --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
      -- bash tools/env/_py14.sh jprune_b.py dump --seed $S --point $P ;;
  score)
    for K in snap primary attach; do                                            # the three judged read-outs; B's plain graph = numbers.base of any of the cards; `unsnapped` is a fourth line of `jprune_b.py commands` (E312 style, not needed by the rule)
      CMD=$(bash tools/env/_py14.sh jprune_b.py commands --seed $S --point $P 2>/dev/null | grep -A1 "^# $P $K\$" | tail -1)
      python3 ../autoresearch/tools/run.py --id E352.s0_${P}_${K}_s$S --seed $S --ram-mb 3000 --retries 0 --require-card --split val --hyp "$HYP" --pred "$PRED" --rule "$RULE" --after E352.s0_${P}_s$S \
        -- $(eval echo "$CMD")
    done ;;
esac
