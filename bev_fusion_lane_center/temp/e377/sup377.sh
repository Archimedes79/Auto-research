#!/bin/bash
# E377 (card v2.1) supervisor: ONE job family on slot 2: trains n_s0 then n_s1 (no early stop), relaunching a run.py id after a crash (resume from the
# 500-step checkpoint), each followed by its GPU dumps on the same slot (blocking) and its CPU read-out in the background.
# usage: bash temp/e377/sup377.sh      (timeout 55 min per the card; head seed = S + 2 as E373)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e377/chain377.log
say() { echo "$(date '+%m-%d %H:%M:%S') [sup377] $*" >> $L; }
source temp/e377/texts377.sh
exec 8>/tmp/e377_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
say "supervisor starts"
for S in 0 1; do
  A=n_s$S; H=$((S+2)); NAME=e377_$A; n=0
  while ! grep -q "^[0-9:]* done $NAME\$" runs/$NAME/log.txt 2>/dev/null; do
    if pgrep -f "run.py --id E377.$A " > /dev/null; then sleep 30; continue; fi
    n=$((n+1)); [ $n -gt 12 ] && { say "$A: 12 launches, giving up"; exit 1; }
    say "$A launch $n (cache seed $S, head seed $H, --none-weight 0.25 --none-weight-near 0.73)"
    python3 ../autoresearch/tools/run.py --id E377.$A --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 55 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jtoken_train.py --name $NAME --seed $S --head-seed $H --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 --none-weight-near 0.73 >> temp/e377/train_$A.out 2>&1
    sleep 20
  done
  say "$A training done"
  bash temp/e377/read377.sh dump $A >> temp/e377/read_$A.out 2>&1
  bash temp/e377/read377.sh cpu $A >> temp/e377/read_$A.out 2>&1 &
done
wait
say "both trainings and read-outs done"
