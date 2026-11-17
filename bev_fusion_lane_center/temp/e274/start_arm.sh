#!/bin/bash
# E274: one arm of one seed through the launcher.  usage: start_arm.sh r|p SEED [--after ID]
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; shift 2
HYP="E274: the E273 pyramid head with recall weights (positives of the node existence loss) at event cells and inside intersections (arm r) and a dual-softmax loss over the long links (arm p = r + it) raises node recall and the topology of the delivered graph: topo F1 above the E273 checkpoints by more than 0.02 on both seeds with the assignment gain kept"
PRED="r: topo F1 0.596 -> about 0.65 (+0.05), node recall 0.83 -> 0.90 at precision >= 0.80, F1@1.0 -> 0.86; p: junction F1 +0.06 over r, long-link precision 0.18 -> 0.30"
RULE="E274 card section 0: KEEP if topo F1 gain over c > +0.02 on both seeds (band 0.02), guards lane_acc_exact p-g >= -0.01, lane_acc_err p-g >= +0.03, F1@1.0 not below c - 0.01; p vs r on junction F1 band 0.03; two seeds, VAL 873, final checkpoint; by hand (verdict274) because card.validate cannot read assignment numbers"
case $ARM in
  r) ID=E274.r_s$S; NAME=e274_r_s$S; ARGS="--node-w-evt 2 --node-w-inter 1 --gate-phases 3 --gate-node-prec 0.75";;
  p) ID=E274.p_s$S; NAME=e274_p_s$S; ARGS="--node-w-evt 2 --node-w-inter 1 --long-match 0.5 --phases 4 5 6 7 --resume-from $PWD/runs/e274_r_s${S}_p3/model.pt --gate-phases 3 --gate-node-prec 0.75";;
  *) echo "arm r or p"; exit 2;;
esac
python3 ../autoresearch/tools/run.py --id $ID "$@" --seed $S --gpu --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
  -- bash tools/env/_pyr.sh qgraph_train.py --name $NAME --seed $S --workers 3 $ARGS
