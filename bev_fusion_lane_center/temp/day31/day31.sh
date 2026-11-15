#!/bin/bash
# Day chain 31 (2026-11-15, lane experiments chat): card E356 (node spacing), CPU only, ONE serial stream at nice 19 (the E354 pair stage is host bound: at most ~4 busy threads from the lane side), every job a run.py job (hard rule 3).
#   spacing count + control (the tagged-score path reproduces the stored snapped card)  ->  part A: radius 2.0 / 2.5 x seeds, snapped  ->  part B: the E346 arms l25 and l31 (full 873 frames)  ->  the table (temp/e356/e356_table.py)
# Stages are idempotent: temp/day31/<stage>.done is skipped; a failed stage stops the chain (log temp/day31/day31.log). Dry run: DAY31_DRY=1 bash temp/day31/day31.sh. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
D=temp/day31
[ -n "$DAY31_DRY" ] && D=temp/day31/dry_$(date +%H%M%S)
L=$D/day31.log
mkdir -p $D
exec 9>/tmp/day31${DAY31_DRY:+_dry}.lock
flock -n 9 || { echo "day31 already runs $(date)" >> $L; exit 0; }
say() { echo "$* $(date)" >> $L; }
HYP0="E356 (card experiments/E356_node-spacing, section 0): "
PRED="E356 card section 1 (prediction with hit windows): radius 2.0: E324 s snapped topo F1 -0.002 (window -0.006 .. +0.002), vert100_dp -0.3 (window -1.0 .. 0.0); radius 2.5: -0.010 (window -0.025 .. -0.002), vert100_dp -1.5 (window -3.0 .. -0.5); l25 topo F1 -0.030 against g1 (window -0.045 .. -0.020), l31 -0.045 (window -0.065 .. -0.030)"
RULE="E356 card section 0 (by hand: replay of the cards, temp/e356/e356_table.py): a radius is taken when topo F1 loses at most 0.01 on both seeds and all four numbers sets with the guards held AND vert100_dp of E324 s snapped falls by at least 0.5 (mean of the seeds); part B: the E346 rule (loss against g1 at most 0.03 on topo F1, lanes found, F1@1.0)"
rp() { # <id> <seed> <command ...>  (CPU job, nice 19)
  local id=$1 seed=$2; shift 2
  if [ -n "$DAY31_DRY" ]; then echo "DRY run.py --id $id --seed $seed -- nice -n 19 $*" >> $L; return 0; fi
  python3 ../autoresearch/tools/run.py --id $id --seed $seed --retries 0 --require-card --split val --timeout 120 --hyp "${HYP0}$id" --pred "$PRED" --rule "$RULE" -- nice -n 19 "$@"
}
stage() { local n=$1 f=$2; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if $f; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
SC="bash tools/env/_py14.sh jends_eval.py score"
chk() { rp E356.a15chk_snap_s0 0 $SC --run e324_s_s0 --seed 0 --spec close:1.5@0.5 --tag c15 --snap --thr 0.1 || return 1; [ -n "$DAY31_DRY" ] && return 0; python3 - <<'PYEOF'
import json, sys
a = json.load(open("runs/e324_s_s0/ends_card_snap_thr0.1.json"))["numbers"]
b = json.load(open("runs/e324_s_s0/ends_card_c15_snap_thr0.1.json"))["numbers"]
d = max(abs(a[w][k] - b[w][k]) for w in ("base", "head") for k in ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "dup_frac"))
print("control recomputation: largest difference", d)
sys.exit(0 if d <= 2e-4 else 1)
PYEOF
}
stage control chk
radius() { # <radius 20|25> <seed>: the snapped read-out (B plain = numbers.base of the same card)
  local R=$1 S=$2 X; X=$([ "$R" = 20 ] && echo 2.0 || echo 2.5)
  rp E356.a${R}_snap_s$S $S $SC --run e324_s_s$S --seed $S --spec close:$X@0.5 --tag c$R --snap --thr 0.1
}
nnsp() { rp E356.nn_spacing 0 bash tools/env/_py14.sh nn_spacing.py; }
stage spacing nnsp
for R in 25 20; do for S in 0 1; do stage a${R}_s$S "radius $R $S"; done; done
armb() { rp E356.$1 0 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash tools/env/_py14.sh e346_arm.py $1; }
stage l25 "armb l25"
stage l31 "armb l31"
touch $D/DAY31_DONE
say "done"
