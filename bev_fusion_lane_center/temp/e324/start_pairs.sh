#!/bin/bash
# E324 arm s: the E320 recipe at 6000 pair steps (the lever of E316), no new code; through the launcher with the card of experiments/E324_e320-duplicates-refined.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E324.s_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The duplicate excess of the E320 recipe is carried by false pairs (E320 delivers +788 / +352 false curves at p >= 0.1, E316 -716 / -1219); a more selective pair stage, the lever of E316 (the pair stage twice as long), applied to the E320 recipe removes part of the guard excess while the attached ends keep the geometry." \
  --pred "read against E316 (snapped, 6000 steps): dup_frac of the PRIMARY read-out over E316 falls from +0.0253 / +0.0119 (E320 at 3000 steps) to +0.013 / +0.006; topo F1 over E316 -0.002 / +0.003; against E312: topo F1 +0.0051 / +0.0099, dup_frac +0.0115 / +0.0068 (seed 0 marginal against the 0.01 guard)" \
  --rule "E324 card section 1 (the rule of E320 with the CUT branch replaced by D87): by hand (jends_eval.py score --attach --snap --snap-nogeom --thr 0.1); KEEP when on both seeds topo F1 of the PRIMARY read-out >= E312 snap-nogeom + 0.005 with the guards held (F1@1.0, dup_frac +0.01 against E312 snapped, vertices, junction F1); TIE kept and refined (D87) when a bound or guard is missed; CUT only by the kill (topo F1 below the floor u' on either seed, loss not finite); VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e324_s_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach --workers 3
