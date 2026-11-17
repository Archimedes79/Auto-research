#!/bin/bash
# E286: the junction maps head on the frozen pyramid of B, through the launcher with the card of experiments/E286_junction-pixel-maps.  usage: start_jmaps.sh <j0|j1|j2> <seed> [smoke]
cd "$(dirname "$0")/../.."
ARM=$1; S=$2
EXTRA=""
case $ARM in j0) CTX=none;; j1) CTX=dil;; j2) CTX=attn;; jj) CTX=${JJ_CTX:-dil}; EXTRA="--joint";; *) echo "arm?"; exit 2;; esac
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E286.smoke_${ARM}_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jmaps_train.py --name e286_smoke_${ARM}_s$S --seed $S --ctx $CTX $EXTRA --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E286.${ARM}_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A junction maps head (1 m: zone, conflict core, entry / exit heat with heading; 0.5 m: stop lines, crossings) on the frozen pixel stage of the lane head B, trained on the junction GT of E285, finds junction entries and exits better than B's own stop / exit event heat maps read the same way (context depth: ${CTX})." \
  --pred "kp_f1 (entry / exit F1 at 3 m, thr 0.3, distance only) +0.15 over the event heat maps of B for the dilated arm; the none / attn arms within 0.05 of it" \
  --rule "E286 card section 0: by hand with evaluation/jmaps_eval.py score, VAL 873, final checkpoint; KEEP above +0.05 over c on both seeds, TIE inside +-0.05 (kept and refined, D82), CUT below -0.05 on both; base for the next step = best mean kp_f1; no TEST" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e286_${ARM}_s$S --seed $S --ctx $CTX $EXTRA --steps 3000 --workers 3
