#!/bin/bash
# E389 (D128): the training arms of the card through run.py (the card's own words), two GPU slots at once: pair 1 = z_s0 + m_s0, pair 2 = z_s1 + m_s1, then the twins z2_s0 + m2_s0 (init seed 1), then, only when G=1, the geometry-only controls zg_s0 + mg_s0.
# usage: TO=30 [STEPS=3000] [G=0] bash temp/e389/e389_train.sh      log: temp/e389/chain.log, per run temp/e389/train_<arm>.out ; the runs resume (--retries 2: a relaunch after a crash resumes from the 500-step checkpoint)
# TO (minutes) comes from the launch gate: steps x the pessimistic s/step x 1.3.  Never edit this file while it runs (bash reads a script by offset).
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
eval "$(python3 temp/e389/e389_texts.py)"
L=temp/e389/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e389 train] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
TO=${TO:-30}
STEPS=${STEPS:-3000}
COMMON="--ident e292_a --steps $STEPS --bs 8 --lr 1e-3 --workers 4"
launch() {   # launch <arm> <seed> <init seed or -> <inputs> <tag>
  local ARM=$1 S=$2 INIT=$3 INP=$4 TAG=$5
  local EXTRA=""
  [ "$INIT" != "-" ] && EXTRA="--init-seed $INIT"
  say "start $TAG (arm $ARM seed $S inputs $INP $EXTRA, timeout $TO min)"
  $RUN --id E389.$TAG --seed $S --gpu --ram-mb 6000 --retries 2 --require-card --split val --timeout $TO --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jroute_train.py --name e389_$TAG --seed $S $EXTRA --arm $ARM --inputs $INP $COMMON >> temp/e389/train_$TAG.out 2>&1
  say "$TAG ended (exit $?)"
}
launch z 0 - full z_s0 &
launch m 0 - full m_s0 &
wait
launch z 1 - full z_s1 &
launch m 1 - full m_s1 &
wait
launch z 0 1 full z2_s0 &
launch m 0 1 full m2_s0 &
wait
if [ "${G:-0}" = "1" ]; then
  launch z 0 - geo zg_s0 &
  launch m 0 - geo mg_s0 &
  wait
fi
say "all arms ended"
