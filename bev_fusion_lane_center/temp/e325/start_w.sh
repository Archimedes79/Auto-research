#!/bin/bash
# E325 arm w: the learned wiring head, one seed through the launcher (GPU slot: the TRAIN records are a live pass of B and the pair run, then the heads train on the CPU in minutes).  usage: start_w.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
python3 ../autoresearch/tools/run.py --id E325.w_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A small softmax head over the delivered piece PLACES in reach (15 features, 545 parameters), trained on TRAIN frames with the oracle label (the rule applied at the matched GT end) and a heavy-tailed keypoint jitter, replaces the nearest-piece rule at the read-out of E312 at equal topology: the one rule step of the delivered graph that had no differentiable replacement" \
  --pred "w - c = 0.000 topo F1 (an imitation of a label that ties the rule); place-level agreement with the oracle label on the real choices: rule 0.750 / 0.745, head 0.790 / 0.785 (the VAL cross-validation diagnostic gave 0.772 / 0.775, distance-only 0.758 / 0.755); lanes found, F1@1.0, dup_frac, junction F1 within 0.003 of c" \
  --rule "E325 card section 0 (by hand: replay; KEEP when snapped topo F1 of w >= c - 0.005 on both seeds with the guards held, TIE kept and refined (D87) otherwise, CUT only by the kill: the head does not learn)" \
  -- bash tools/env/_pyr.sh jwire_train.py run --seed $S --name e325_w_s$S --heads 3 --epochs 60 --workers 3
