#!/bin/bash
# Day chain 29 (2026-11-15, lane experiments chat): card E353 (paint branch at 25 % hidden width), six GPU jobs and their read-outs, every job a run.py job (hard rule 3).
#   1 wait for the quiet-host timing E351.q_eager (status ok) or for DAY29_WAIT_UNTIL (HHMM, default 1930)   2 the step-0 rows n1z (CPU, --steps 0) and the fine-tunes cp / n1, seed 0 and seed 1 in parallel (two GPU slots)
#   3 per run: dump (GPU gate, one at a time), the four cards snap / primary / attach / unsnapped (CPU, two streams)   4 the rule replay (jrule_removal.py against c and against cp)
# Stages are idempotent: a stage with temp/day29/<stage>.done is skipped; a failed stage stops the chain (log temp/day29/day29.log). Dry run: DAY29_DRY=1 bash temp/day29/day29.sh (nothing is run).
# Sunday: no daytime GPU limit (D100); the can_start window of chain_lib.sh is NOT used on purpose (see the lead's message of 2026-11-15 ~14:15). Never edit this script while it runs.
cd "$(dirname "$0")/../.."
D=temp/day29
[ -n "$DAY29_DRY" ] && D=temp/day29/dry_$(date +%H%M%S)
L=$D/day29.log
mkdir -p $D
exec 9>/tmp/day29${DAY29_DRY:+_dry}.lock
flock -n 9 || { echo "day29 already runs $(date)" >> $L; exit 0; }
say() { echo "$* $(date)" >> $L; }
CARD=experiments/E353_paint-branch-width-25
HYP0="E353 (card $CARD, section 0): "
RULE0="E353 card section 0 (by hand: replay with jrule_removal.py against c = E324 s of the same seed and against cp = the continuation): ADOPT when the topo F1 loss is at most 0.01 on both seeds and all three read-outs with the guards held, against c and against cp"
okrun() { python3 -c "
import json,sys
try: st=json.load(open('experiment_execution/runs/$1/run.json')).get('status')
except Exception: st=None
sys.exit(0 if st=='ok' else 1)"; }
# rp <id> <seed> <gpu 0|1> <command ...>: one run.py job (blocks until it ends)
rp() {
  local id=$1 seed=$2 g=$3; shift 3
  local fl=""; [ "$g" = 1 ] && fl="--gpu --ram-mb 6000"
  if [ -n "$DAY29_DRY" ]; then echo "DRY run.py --id $id --seed $seed $fl -- $*" >> $L; return 0; fi
  python3 ../autoresearch/tools/run.py --id $id --seed $seed $fl --retries 0 --require-card --split val --timeout 240 \
    --hyp "${HYP0}$id" --pred "E353 card section 1 (prediction with hit windows): snapped topo F1 of n1 minus c -0.002, PRIMARY -0.002, attach-only -0.003; cp minus c +0.001; n1z (no fine-tune) -0.004 snapped" --rule "$RULE0" -- "$@"
}
stage() { local n=$1 f=$2; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if $f; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
both() { local f=$1 p0 p1 r=0; $f 0 & p0=$!; $f 1 & p1=$!; wait $p0 || r=1; wait $p1 || r=1; return $r; }
SPEC() { echo $CARD/spec_n1_s$1.json; }
FT="--steps 1500 --lr 3e-4 --e2e --scales --maps-w 1.0 --geom-w 0.5 --attach --workers 4"

w0() { # wait for the quiet-host timing (it needs both slots idle and a quiet host) or for the time limit
  local until=${DAY29_WAIT_UNTIL:-1930}
  [ -n "$DAY29_DRY" ] && return 0
  until okrun E351.q_eager || [ "$(date +%H%M)" -ge "$until" ]; do sleep 30; done
  okrun E351.q_eager && say "E351.q_eager ok" || say "E351.q_eager not done at $until: go on"
  return 0
}
stage wait w0

z() { rp E353.n1z_s$1 $1 0 env CUDA_VISIBLE_DEVICES= BFLC_SIG=1.0 bash tools/env/_py14.sh jnarrow_train.py --name e353_n1z_s$1 --init runs/e324_s_s$1/model.pt --seed $1 --narrow $(SPEC $1) --steps 0 --e2e --scales --maps-w 1.0 --geom-w 0.5 --attach --workers 2; }
stage n1z_all "both z"
cp_run() { rp E353.cp_s$1 $1 1 env BFLC_SIG=1.0 bash tools/env/_pyr.sh jnarrow_train.py --name e353_cp_s$1 --init runs/e324_s_s$1/model.pt --seed $1 $FT; }
n1_run() { rp E353.n1_s$1 $1 1 env BFLC_SIG=1.0 bash tools/env/_pyr.sh jnarrow_train.py --name e353_n1_s$1 --init runs/e324_s_s$1/model.pt --seed $1 --narrow $(SPEC $1) $FT; }
train_s() { local S=$1; cp_run $S & local a=$!; sleep 30; n1_run $S & local b=$!; wait $a || return 1; wait $b || return 1; }
stage train_s0 "train_s 0"
stage train_s1 "train_s 1"

dump() { rp E353.dump_$1 $2 1 flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jnarrow_train.py dump --run e353_$1; }
cards() { # <arm> <seed>: the four cards of a run, two streams
  local a=$1 S=$2 R=e353_$1
  ( rp E353.card_${a}_snap $S 0 bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1
    rp E353.card_${a}_attach $S 0 bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --thr 0.1 ) &
  ( rp E353.card_${a}_primary $S 0 bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --snap --snap-nogeom --thr 0.1
    rp E353.card_${a}_unsnapped $S 0 bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --thr 0.1 ) &
  wait
}
post() { # <arm>_s<S>
  local r=$1; local a=${r%_s*} S=${r##*_s}
  dump $r $S || return 1
  cards ${a}_s$S $S
}
post_z() { local S=$1; dump n1z_s$S $S && rp E353.card_n1z_s${S}_snap $S 0 bash tools/env/_py14.sh jends_eval.py score --run e353_n1z_s$S --seed $S --snap --thr 0.1 && rp E353.card_n1z_s${S}_primary $S 0 bash tools/env/_py14.sh jends_eval.py score --run e353_n1z_s$S --seed $S --attach --snap --snap-nogeom --thr 0.1; }
for r in n1_s0 cp_s0 n1_s1 cp_s1; do
  # the card ids: E353.card_<arm>_s<S>_<readout> (the run name carries the seed)
  stage post_$r "post $r"
done
# the step-0 rows: dump + snapped / primary cards only (report rows)
for S in 0 1; do stage post_n1z_s$S "post_z $S"; done

rule() {
  rp E353.rule_c 0 0 bash tools/env/_py14.sh jrule_removal.py --arm e353_n1_s --control e324_s_s --readouts snap primary attach
  rp E353.rule_cp 0 0 bash tools/env/_py14.sh jrule_removal.py --arm e353_n1_s --control e353_cp_s --readouts snap primary attach
}
stage rule rule
touch $D/DAY29_DONE
say "done"
