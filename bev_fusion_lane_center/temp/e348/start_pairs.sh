#!/bin/bash
# E348 (card experiments/E348_e324s-recipe-without-paint-branch): the E324 arm s recipe (E320 recipe at 6000 pair steps: --e2e --scales --maps-w 1.0 --geom-w 0.5 --attach) on the maps head of E332 arm a (no 0.5 m branch, width 64),
# through the launcher with the card. usage: start_pairs.sh <seed>. D101: --workers 4; D100: no nice; no --cache (E349 is opt-in and not adopted).
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e332_ma_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E348.w_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val \
  --hyp "The 0.5 m paint branch is not needed on the system of record either: the E324 s recipe (6000 pair steps, geometry loss, attachment) on the maps head without that branch, evidence with its four non-paint inputs, loses at most 0.01 topo F1 against E324 s of the same seed on the snapped, the PRIMARY and the attach read-out." \
  --pred "topo F1 -0.003 against E324 s of the same seed on the snapped and the PRIMARY read-out (0.6951 / 0.6980 and 0.6882 / 0.6918), -0.005 on the attach read-out (0.6566 / 0.6596); the maps head needs 1.78 instead of 3.26 GMACs at the same width" \
  --rule "E348 card section 0: evaluation/jrule_removal.py --arm e348_w_s --control e324_s_s --readouts snap primary attach: ADOPT when topo F1 loses at most 0.01 against E324 s of the same seed on both seeds and all three read-outs with the guards held (F1@1.0 <= -0.01, dup_frac <= +0.01, vertices <= +1.0, junction F1 <= -0.03); label ADOPT-FIRM when the paired 95 % lower bound is >= -0.010 on every seed and read-out, else ADOPT-POINT; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e348_w_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach --workers 4
