#!/bin/bash
# E387: the two f arms (card section 0) in the two GPU slots at once, 8000 steps each; the --hyp / --pred / --rule strings are the CARD's own words (texts387.sh, regenerated from the FINAL card by `jarm.card_texts`). The --timeout (minutes) comes from the gate: steps x the pessimistic s/step x 1.3, passed in TO (default 75).
# usage: TO=75 bash temp/e387/train.sh      log: temp/e387/chain.log ; the runs resume (--retries 2: a relaunch after a crash resumes from the 500-step checkpoint)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e387/texts387.sh
L=temp/e387/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e387 train] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
TO=${TO:-75}
COMMON="--ident e292_a --steps 8000 --bs 4 --seq 3 --lr 1e-3 --workers 4 --block fused"
say "start f_s0 and f_s1 (timeout $TO min)"
for S in 0 1; do
  $RUN --id E387.f_s$S --seed $S --gpu --ram-mb 8000 --retries 2 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jpast_train.py --name e387_f_s$S --seed $S $COMMON >> temp/e387/train_f_s$S.out 2>&1 &
done
wait
say "f_s0 and f_s1 ended"
