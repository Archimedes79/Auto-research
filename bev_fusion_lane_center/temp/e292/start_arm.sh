#!/bin/bash
# E292: the fine-tune of B's FINAL checkpoint with the pyramid frozen (phases 3 4 5 6 9), arm c2 (no layer, the matched reference) or a (the one-pass Matrix-NMS layer R 1.75 m, sigma 0.1 after the node head).
# usage: start_arm.sh <c2|a> <seed>      (through the launcher; the card is experiments/E292_matrix-nms-layer/experiment.md, run ids E292.c2_s<S> / E292.a_s<S>)
cd "$(dirname "$0")/../.."
ARM=$1; S=$2
case $ARM in
  c2) EXTRA=""; HYP="E292 matched reference: B's final checkpoint fine-tuned with the schedule of the card (phases 3 4 5 6 9, pyramid frozen) WITHOUT the layer, read out with the hard close-node step, costs less than the band against E312 (precondition of the comparison)";
      PRED="c2 topo F1 within -0.01 of c (E312 snapped 0.6825 / 0.6864) on both seeds with the guards held; not predicted more exactly (that is what c2 measures)";;
  a)  EXTRA="--soft-nms-mnms 1.75 0.1"; HYP="E292: the one-pass Matrix-NMS layer after the node head, trained end to end in the fine-tune, delivers a graph that read out WITHOUT the hard close-node step and wired with E312's connectors scores within the band of the same fine-tune without the layer";
      PRED="a - c: snapped topo F1 -0.003, lanes found -0.009, junction F1 +0.005, F1@1.0 -0.003, vertices -0.7 per 100 m, dup_frac -0.003 (card section 1)";;
  *) echo "arm c2 or a"; exit 2;;
esac
RULE="E292 card section 0: by hand (replay of the stored cards: jends_eval.py score --snap --thr 0.1 --full --base --spec --tag); a seed PASSES when topo F1 of a >= c2 - 0.01 and >= c - 0.01 with the guards held; PRECONDITION c2 passes against c; KEEP / TIE (kept and refined, D87) / CUT only by the kill (card section 0)"
B=$(pwd)/runs/e273_p_s$S/model.pt
python3 ../autoresearch/tools/run.py --id E292.${ARM}_s$S --seed $S --gpu --retries 0 --require-card --split val --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
  -- bash tools/env/_pyr.sh qgraph_train.py --name e292_${ARM}_s$S --seed $S --workers 3 --phases 3 4 5 6 9 --steps 3000 2500 1000 750 1000 --resume-from $B --gate-phases 3 --gate-node-prec 0.75 --gate-node-rec 0.5 $EXTRA
