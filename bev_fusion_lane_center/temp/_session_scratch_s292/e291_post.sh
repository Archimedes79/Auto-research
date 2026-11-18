#!/bin/bash
# E291 post-processing of one seed S: wait for the twin c and the arm b (E291.c_s<S>, E291.b_s<S> ok), the dumps (GPU jobs, one after the other), then the cards (CPU, two streams).
# usage: e291_post.sh <seed>      (arm a is not run today: card section 4 caveat 4)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
S=$1
LOG=temp/_session_scratch_s292/logs/e291_post_s$S.log
say() { echo "$* $(date +%H:%M:%S)" >> $LOG; }
okrun() { python3 -c "
import json,sys
try: st=json.load(open('experiment_execution/runs/$1/run.json')).get('status')
except Exception: st=None
sys.exit(0 if st=='ok' else 1)"; }
for R in E291.c_s$S E291.b_s$S; do
  n=0
  until okrun $R; do
    n=$((n+30)); [ $n -gt 28800 ] && { say "$R not ok after 8 h"; exit 2; }
    sleep 30
  done
  say "$R ok"
done
HYP0="E291 read-out of the lane-count prior (card experiments/E291_sd-road-level-lane-counts, section 0): "
RULE0="E291 card section 0 (by hand: replay, decide.py is not used): G / R / C from the pooled paired scene bootstrap of snapped topo F1 against the recipe twin; kill (5) from the s / sc read-outs; the clean row is an upper bound (the table comes from GT end clusters)"
dump() { # id, extra args
  python3 ../autoresearch/tools/run.py --id $1 --seed $S --gpu --retries 1 --require-card --split val --timeout 120 --hyp "${HYP0}dump of $2" \
    --pred "the read-outs of one model from one pass: keypoints identical across read-outs, only the prior term differs" --rule "$RULE0" -- bash temp/e291/run_staged_gpu.sh jends_eval.py dump --run $2 $3
  say "$1 exit $?"
}
dump E291.dump_b_s$S e291_b_s$S "--prior-mode clean,noise10,noise30,pm1,u,s,sc"
dump E291.dump_c_s$S e291_c_s$S ""
card() { # id, run, extra flags
  python3 ../autoresearch/tools/run.py --id $1 --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "${HYP0}card $1" \
    --pred "snapped topo F1 of the clean read-out about +0.008 over the twin, -0.003 at 30 % wrong counts (card prediction)" --rule "$RULE0" -- bash tools/env/_py14.sh jends_eval.py score --run $2 --seed $S --thr 0.1 $3
  say "$1 exit $?"
}
streamA() {
  card E291.card_b_s${S}_clean e291_b_s${S}_clean "--snap --tag clean"
  card E291.card_b_s${S}_u e291_b_s${S}_u "--snap --tag u"
  card E291.card_b_s${S}_s e291_b_s${S}_s "--snap --tag s"
  card E291.card_c_s${S}_snap e291_c_s$S "--snap"
  card E291.card_b_s${S}_noise10 e291_b_s${S}_noise10 "--snap --tag noise10"
}
streamB() {
  card E291.card_b_s${S}_noise30 e291_b_s${S}_noise30 "--snap --tag noise30"
  card E291.card_b_s${S}_sc e291_b_s${S}_sc "--snap --tag sc"
  card E291.card_b_s${S}_clean_unsnapped e291_b_s${S}_clean "--tag clean"
  card E291.card_c_s${S}_unsnapped e291_c_s$S ""
  card E291.card_b_s${S}_pm1 e291_b_s${S}_pm1 "--snap --tag pm1"
}
streamA & streamB & wait
say "e291 post seed $S done"
