#!/bin/bash
# E363: the ten CPU scoring jobs through run.py (two chains, one per seed: c -> t -> a -> cw -> aw, --after), nice, 2 threads, --retries 0 (the scorer refuses to overwrite a tagged card: a failed job needs a NEW tag, never a retry).
#   wsl bash temp/e363/start_chain.sh [dry]     ('dry' as an ARGUMENT: an env flag does not pass from Git Bash through wsl)
cd "$(dirname "$0")/../.."
R="$(pwd)"
MODE=$1
job() {   # job <id> <seed> <arm tag> <graph c2|a> <staged 0|1> <spec> <extra flags>
  local id=$1 S=$2 tag=$3 g=$4 staged=$5 spec=$6 extra=$7
  local after=""; [ -n "$PREV" ] && after="--after $PREV"
  local full="$R/temp/e292/predfull_e292_${g}_s$S.pkl" base="$R/runs/e292_${g}_s$S/model.pt" run="e363_${g}_s$S"
  [ -f "$full" ] && [ -f "$base" ] && [ -f "runs/$run/ends_val.pkl" ] || { echo "missing $full or $base or runs/$run/ends_val.pkl"; exit 3; }
  local runner="bash tools/env/_py14.sh"; [ "$staged" = "1" ] && runner="bash temp/e358/run_staged.sh"
  local cmd="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $runner jends_eval.py score --run $run --seed $S --thr 0.1 --attach --snap --snap-nogeom --full $full --base $base --spec $spec $extra --tag $tag"
  local line=(python3 ../autoresearch/tools/run.py --id $id --seed $S --require-card --retries 0 --split val $after
    --hyp "E363: the end-to-end stack on faithful graphs: the pair run of E324 s with the node head of B' (c2 plain / a with the Matrix-NMS layer), end-to-end read-out, scoring job $id"
    --pred "a - c topo F1 +0.016 seed 0 / +0.024 seed 1 (hit +-0.008); aw - a +0.0019 / +0.0016 (hit +-0.003); expected token TIE"
    --rule "E363 card section 0 (by hand: D87 replay by temp/e363/verdict_e363.py; KEEP = candidate only, nothing adopted)"
    -- $cmd)
  if [ "$MODE" = "dry" ]; then printf '%q ' "${line[@]}"; echo; else "${line[@]}" > "temp/e363/launch_$id.log" 2>&1 & sleep 3; fi
  PREV=$id
}
for S in 0 1; do
  PREV=""
  job E363.c_s$S  $S e363c  c2 0 plain@0.5 ""
  job E363.t_s$S  $S e363t  c2 0 close:1.5@0.5 ""
  job E363.a_s$S  $S e363a  a  0 plain@0.5 ""
  job E363.cw_s$S $S e363cw c2 1 plain@0.5 "--wire head --wire-run e358_w_s$S"
  job E363.aw_s$S $S e363aw a  1 plain@0.5 "--wire head --wire-run e358_w_s$S"
done
wait
