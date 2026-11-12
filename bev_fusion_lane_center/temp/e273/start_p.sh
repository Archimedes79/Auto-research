#!/bin/bash
# E273 lane head, one seed through the launcher (all phases, a gate after every phase). Seed 1 waits for seed 0.
cd "$(dirname "$0")/../.."
S=$1
AFTER=""
[ "$S" != "0" ] && AFTER="--after E273.p_s0"
python3 ../autoresearch/tools/run.py --id E273.p_s$S $AFTER --seed $S --gpu --retries 0 --require-card \
  --hyp "The lane head (pixel maps -> grid-cell nodes -> Hermite links -> soft sections -> map assignment) as one differentiable function assigns map pieces to the delivered lane nodes better than the geometric assignment of the E272 chain when the map is 1-2 m off, and its delivered graph is not worse than recall_sp0_dp in topo F1 and F1@1.0 while its junction F1 is higher" \
  --pred "lane_acc_err p - g = +0.10; junction F1 0.27 -> 0.45; topo F1 level with d (0.740); F1@1.0 level with d (0.878)" \
  --rule "E273 card section 0 (band 0.03; guards lane_acc_exact not below g - 0.01, topo_F1 and F1@1.0 not below d - 0.03), two seeds, VAL 873 frames, final checkpoint, gates per phase from the card kill criterion; by hand because card.validate cannot read assignment numbers" \
  -- bash tools/env/_pyr.sh qgraph_train.py --name e273_p_s$S --seed $S --workers 3
