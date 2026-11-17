#!/bin/bash
# E360 (card experiments/E360_matrix-nms-in-the-e2e-stack): the nine CPU scoring jobs through run.py, chained with --after, one at a time, nice, 2 threads, --retries 0
# (the scorer refuses to overwrite a tagged card: a failed job needs a NEW tag, never a retry).   wsl bash temp/e360/start_chain.sh        E360_DRY=1 prints the run.py lines only
cd "$(dirname "$0")/../.." || exit 2
R="$(pwd)"
RUN="python3 ../autoresearch/tools/run.py"
PREV=""
job() {   # job <id> <seed> <arm tag> <graph: c2|a> <spec> <flags>
  local id=$1 S=$2 tag=$3 g=$4 spec=$5 flags=$6
  local after=""
  [ -n "$PREV" ] && after="--after $PREV"
  local full="$R/temp/e292/predfull_e292_${g}_s$S.pkl" base="$R/runs/e292_${g}_s$S/model.pt"
  [ -f "$full" ] && [ -f "$base" ] || { echo "missing $full or $base"; exit 3; }
  local cmd="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jends_eval.py score --run e324_s_s$S --seed $S --thr 0.1 $flags --full $full --base $base --spec $spec --tag $tag"
  local line="$RUN --id $id --hyp \"E360: the Matrix-NMS layer (E292 a) read out at the end-to-end read-out with the pair run of E324 s, no hard step, is better than the graph without layer and without step (E292 c2 plain): scoring job $id\" --pred \"a - c topo F1 +0.011 seed 0 / +0.019 seed 1 (hit +-0.010); expected token TIE\" --rule \"E360 card section 0 (by hand: D87 replay by temp/e360/verdict_e360.py; KEEP = candidate only, nothing adopted)\" --split val --seed $S --require-card --retries 0 $after -- $cmd"
  if [ "$E360_DRY" = "1" ]; then echo "$line"; else eval "$line" > "temp/e360/launch_$id.log" 2>&1 & sleep 3; fi
  PREV=$id
}
# seed 0: the drift check first, then the arms, then the own-keypoint witness; seed 1 without t
job E360.t_s0  0 e360t  c2 close:1.5@0.5 "--attach --snap --snap-nogeom"
job E360.c_s0  0 e360c  c2 plain@0.5     "--attach --snap --snap-nogeom"
job E360.a_s0  0 e360a  a  plain@0.5     "--attach --snap --snap-nogeom"
job E360.ck_s0 0 e360ck c2 plain@0.5     "--snap --snap-nogeom"
job E360.ak_s0 0 e360ak a  plain@0.5     "--snap --snap-nogeom"
job E360.c_s1  1 e360c  c2 plain@0.5     "--attach --snap --snap-nogeom"
job E360.a_s1  1 e360a  a  plain@0.5     "--attach --snap --snap-nogeom"
job E360.ck_s1 1 e360ck c2 plain@0.5     "--snap --snap-nogeom"
job E360.ak_s1 1 e360ak a  plain@0.5     "--snap --snap-nogeom"
wait
