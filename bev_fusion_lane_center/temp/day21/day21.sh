#!/bin/bash
# Day chain 21 (2026-11-14, lane experiments chat; D99 / card E345 step 0, part A): the pruning sensitivity of E312 and E324 s, one model after the other, the two seeds of a model in parallel.
# Per model (job flow of the docstring of evaluation/jprune_table.py; every GPU job and every CPU job is a run.py job with the card E345, one record each):
#  1 GPU sweep, both seeds  2 CPU plan (seed 0's sweep) -> score_points.json  3 GPU sweep --score-points (the dumps of the planned points), both seeds  4 CPU score of every planned point, both seeds
#  5 CPU plan again (v appears)  6 GPU sweep --score-points (the dump of v), both seeds  7 CPU score of v (--only 120)  8 CPU boot (paired scene bootstrap of every scored point against the null)  then, after both models: 9 table.
# Stages are idempotent: a stage with temp/day21/<stage>.done is skipped, so a restart (after a WSL restart or a failed stage) goes on where it stopped; a failed stage stops the chain (log temp/day21/day21.log).
# GPU starts follow the window rule of chain_lib.sh (can_start); the GPU jobs wait for GPU room by themselves (evaluation/gpu_gate.py, 12 h, then exit 75); run.py hands out its two GPU slots first come first served.
# Dry run (nothing is run, no marker outside temp/day21/dry_*): DAY21_DRY=1 bash temp/day21/day21.sh. Never edit this script while it runs. Marker temp/day21/DAY21_DONE.
cd "$(dirname "$0")/../.."
D=temp/day21
[ -n "$DAY21_DRY" ] && D=temp/day21/dry_$(date +%H%M%S)
L=$D/day21.log
mkdir -p $D temp/e345
exec 9>/tmp/day21${DAY21_DRY:+_dry}.lock
flock -n 9 || { echo "day21 already runs $(date)" >> $L; exit 0; }
[ -f $D/DAY21_DONE ] && { echo "day21 already done $(date)" >> $L; exit 0; }
. temp/e330/chain_lib.sh
if [ -n "$DAY21_DRY" ]; then can_start() { true; }; fi
say() { echo "$* $(date)" >> $L; }

# rp <run id> <seed> <gpu 0|1> <command ...>: one run.py job (blocks until it ends)
rp() {
  local id=$1 seed=$2 g=$3; shift 3
  local fl=""
  [ "$g" = 1 ] && { fl="--gpu"; wait_start; }
  if [ -n "$DAY21_DRY" ]; then echo "DRY run.py --id $id --seed $seed $fl -- $*" >> $L; return 0; fi
  python3 ../autoresearch/tools/run.py --id $id --seed $seed $fl --retries 0 --require-card --split val --timeout 720 \
    --hyp "D99 step 0 (E345 part A): how much of each unit of the stored junction system can be taken away without retraining; job $id" \
    --pred "E345 card section 3: v (all FREE head units) loses 0.005 topo F1 on the judgement half and saves 0.8 GMACs of the heads' 3.278; at least 15 of 45 units have a sweep-FREE point" \
    --rule "E345 card sections 0 and 3, by hand: analysis, selected on half of the VAL scenes and judged on the other half; loss = null minus ablated on the worse of the snapped and the differentiable read-out" \
    -- "$@"
}
# stage <name> <function>: run the function once (marker), stop the chain when it fails
stage() { local n=$1 f=$2; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if $f; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
# both_seeds <function taking a seed>: the two seeds in parallel, fails when one fails
both_seeds() { local f=$1 p0 p1 r=0; $f 0 & p0=$!; $f 1 & p1=$!; wait $p0 || r=1; wait $p1 || r=1; return $r; }

PROBE="bash tools/env/_pyr.sh jprune_probe.py"
TABLE="bash tools/env/_py14.sh jprune_table.py"

pipeline() {   # pipeline <name e312|e324s> <run prefix e312_p_s|e324_s_s>
  local M=$1 RP=$2
  local R0=${RP}0 R1=${RP}1
  s1() { rp E345.a_${M}_s$1 $1 1 $PROBE sweep --run ${RP}$1 --out temp/e345/${RP}$1; }
  s2() { rp E345.p_${M} 0 0 $TABLE plan --sweep temp/e345/$R0/sweep.json --out temp/e345/$R0/score_points.json && { [ -n "$DAY21_DRY" ] || cp temp/e345/$R0/score_points.json temp/e345/$R1/score_points.json; }; }
  s3() { rp E345.d_${M}_s$1 $1 1 $PROBE sweep --run ${RP}$1 --out temp/e345/${RP}$1 --score-points temp/e345/${RP}$1/score_points.json; }
  s4() { rp E345.s_${M}_s$1 $1 0 $TABLE score --run ${RP}$1 --points temp/e345/${RP}$1/score_points.json; }
  s5() { rp E345.p2_${M} 0 0 $TABLE plan --sweep temp/e345/$R0/sweep.json --out temp/e345/$R0/score_points.json && { [ -n "$DAY21_DRY" ] || cp temp/e345/$R0/score_points.json temp/e345/$R1/score_points.json; }; }
  s6() { rp E345.v_${M}_s$1 $1 1 $PROBE sweep --run ${RP}$1 --out temp/e345/${RP}$1 --score-points temp/e345/${RP}$1/score_points.json; }
  s7() { rp E345.sv_${M}_s$1 $1 0 $TABLE score --run ${RP}$1 --points temp/e345/${RP}$1/score_points.json --only 120; }
  s8() { rp E345.b_${M}_s$1 $1 0 $TABLE boot --run ${RP}$1 --points temp/e345/${RP}$1/score_points.json; }
  stage ${M}_1 "both_seeds s1"
  stage ${M}_2 s2
  stage ${M}_3 "both_seeds s3"
  stage ${M}_4 "both_seeds s4"
  stage ${M}_5 s5
  stage ${M}_6 "both_seeds s6"
  stage ${M}_7 "both_seeds s7"
  stage ${M}_8 "both_seeds s8"
}

say "day21 starts (dry: ${DAY21_DRY:-no})"
pipeline e312 e312_p_s
pipeline e324s e324_s_s
s9() { rp E345.t 0 0 $TABLE table --sweeps temp/e345/e312_p_s0,temp/e345/e312_p_s1,temp/e345/e324_s_s0,temp/e345/e324_s_s1 --out experiments/E345_pruning-sensitivity-step0; }
stage table s9
touch $D/DAY21_DONE
say "done"
