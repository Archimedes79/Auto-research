#!/bin/bash
# E347 stage 1: score each arm with the staged maps scorer (CPU, one at a time, run.py job E347.score_<arm>_s<S>) as soon as its training run is ok.
# The control's numbers are in the card BEFORE this script is started (card section 2b (e)); arms scored in the order g0 o0 g1 o1.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for X in "g 0" "o 0" "g 1" "o 1"; do
  set -- $X; A=$1; S=$2
  n=0
  until python3 -c "
import json,sys
try:
    st=json.load(open('experiment_execution/runs/E347.${A}_s${S}/run.json')).get('status')
except Exception:
    st=None
sys.exit(0 if st=='ok' else 1)"; do
    n=$((n+30)); [ $n -gt 14400 ] && { echo "E347.${A}_s${S} not ok after 4 h"; exit 2; }
    sleep 30
  done
  echo "E347.${A}_s${S} ok $(date +%H:%M:%S): scoring"
  python3 ../autoresearch/tools/run.py --id E347.score_${A}_s$S --seed $S --retries 0 --split val --timeout 240 \
    --hyp "E347 stage 1 label (by hand: maps scorer): arm ${A} seed $S read in its rule read-out and in the other one, at the run's own best threshold on the 0.05 - 0.95 grid and at 0.3" \
    --pred "arm g: keypoint F1 at 3 m about 0.02 below the control, crowded recall about 0.10 below; arm o: F1 at 3 m about 0.015 below, at 1.5 m 0.03 below, crowded 0.03 below" \
    --rule "E347 card section 0 stage 1: PASS / FAIL / INCONCLUSIVE clauses at the best threshold of each run (by hand: maps scorer, a label and a kill gate)" \
    -- bash temp/e347/run_staged.sh jmaps_eval.py score --run e347_${A}_s$S --workers 2
  echo "E347.score_${A}_s$S exit $? $(date +%H:%M:%S)"
done
echo "arm scoring done"
