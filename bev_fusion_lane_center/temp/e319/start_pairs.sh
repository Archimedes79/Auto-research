#!/bin/bash
# E319 stage 2: the E312 pair head recipe (--e2e --scales --maps-w 1.0, 3000 steps) on the maps with the exchange, through the launcher with the card of experiments/E319_heading-guided-exchange (revision 2026-11-14).
# usage: start_pairs.sh <seed>.  Only after the gate of stage 1 (temp/e319/kill_check.py: a maps gain over E307). One GPU job at a time by day (D98): called by temp/e319/chain.sh; nice -n 10 as in temp/e341/start_pairs.sh.
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e319_m_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E319.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The junction maps head has an isotropic context; a forward / backward exchange along the lane heading (ahead / behind at 2, 4, 8 m, two rounds, differentiable) gives the ends the evidence of the continuing lane: Mirko's hypothesis that the right shape needs a forward-backward information exchange." \
  --pred "topo F1 +0.005 over E312 of the same seed on the snapped read-out at p >= 0.1 (a TIE is the expected outcome); the maps stage predicted keypoint F1 +0.008 and entry recall beyond 35 m +0.03 over E307" \
  --rule "E319 card section 0: by hand (jmaps_eval.py score for the maps stage; jends_eval.py score --snap --thr 0.1 for the read-out); KEEP when topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept (the head without the block is cheaper); CUT below -0.01 or the kill (stage 1 keypoint F1 below E307 by 0.03); this stage ran because the maps gate was met (keypoint F1 or far entry recall at least +0.01 over E307); VAL 873, no TEST" \
  -- nice -n 10 bash tools/env/_pyr.sh jends_train.py --name e319_p_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --workers 3
