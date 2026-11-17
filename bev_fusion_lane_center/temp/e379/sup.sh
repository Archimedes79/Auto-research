#!/bin/bash
# E379 supervisor (card v2 section 2): the two distilled heads d_s0, d_s1 (the rule-bound runs), one after another on one slot with resume (`tools/maintenance/keep_alive.sh`), each read out as soon as it is trained (dumps on the GPU in the slot, the CPU read-out in the background),
# then the exploratory d4_s0 (--no-card, never a verdict). It does NOT start by itself: it waits for the marker `temp/e379/GO_D`, which the lane chat creates when (i) Fable's flags `--distill-from` / `--distill-weight` have passed their code critic and are on main, (ii) the lead
# gave the slot word. The schedule control l_s0 and the self-teacher ds_s0 are NOT in this script: l only if d passes clause (i) on both seeds (`temp/e379/GO_L`), ds likewise; both are started by hand then. Windows: the GPU night 22:00 - 07:00 and the weekday day window 08:30 - 17:30.
# the lead's yield rule (21:00): if the two parallel trainings show the CUDA unknown error / stall pattern (E362: several resumes in an hour), THIS chain yields: ka_keep gives up after 3 launches (the first + 2 resumes) and the chain stops; nothing of Fable's is paused; resume it by hand after his chain ends.
# usage: bash temp/e379/sup.sh      log: temp/e379/chain.log, training logs temp/e379/train_<name>.out
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source tools/maintenance/keep_alive.sh
ka_init temp/e379/chain.log e379_sup lane_e379_sup
source temp/e379/texts379.sh
RUN="python3 ../autoresearch/tools/run.py"
in_window() { ka_in_window 2200 0700 || ka_in_window 0830 1730; }
say "armed: waits for the marker temp/e379/GO_D and a GPU window"
until [ -f temp/e379/GO_D ] && in_window; do sleep 60; done
grep -q -- "--distill-from" training/train/jtoken_train.py || { say "STOP: --distill-from is not in training/train/jtoken_train.py"; exit 1; }
say "GO"
train() {  # ID NAME SEED HEAD WEIGHT CARDFLAG
  local id=$1 name=$2 seed=$3 head=$4 w=$5 card=$6
  ka_keep "$id" "$name" 3 $RUN --id "$id" --seed "$seed" --gpu --ram-mb 6000 --retries 0 $card --split val --timeout 130 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name "$name" --seed "$seed" --head-seed "$head" --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --distill-from "runs/e373_cb_s$seed,runs/e367_b25_s$seed" --distill-weight "$w" >> "temp/e379/train_$name.out" 2>&1
}
for t in "0 2" "1 3"; do
  set -- $t
  S=$1; H=$2
  train "E379.d_s$S" "e379_d_s$S" "$S" "$H" 1.0 --require-card || exit 1
  bash temp/e379/read379.sh dump "d_s$S" >> "temp/e379/read_d_s$S.out" 2>&1
  (bash temp/e379/read379.sh cpu "d_s$S" >> "temp/e379/read_d_s$S.out" 2>&1 &)
done
train "E379.d4_s0" "e379_d4_s0" 0 2 4.0 --no-card || exit 1
say "d_s0, d_s1 and the exploratory d4_s0 are trained; the read-outs of d_s0 / d_s1 ran (CPU parts in the background); d4_s0 and the controls l / ds are read by hand"
