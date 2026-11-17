#!/bin/bash
# Day chain 17 (2026-11-14, for Sunday night and later): the next arm of the width ladder of the junction maps head, chosen by the E333 verdict (card E339 section 0, card E340): it waits for temp/e333/E333_DONE and reads the LAST line starting with
# "verdict:" of temp/e333/rule.log (evaluation/jrule_removal.py): "verdict: ADOPT ..." -> E340 (plain width 32); anything else (NOT adopted, INCOMPLETE, no file) -> E339 (width 40 taught by the E307 head). The chosen arm needs temp/<exp>/READY + ACCEPTED,
# then runs temp/<exp>/chain.sh (every start goes through chain_lib.sh: window rule and GPU slots). Run-once guard and lock: a second start (WSL restart) after DAY17_DONE does nothing. Never edit this script while it runs.
# Log temp/day17/day17.log, marker temp/day17/DAY17_DONE.
cd "$(dirname "$0")/../.."
L=temp/day17/day17.log; LOGD=temp/day17
mkdir -p $LOGD
[ -f temp/day17/DAY17_DONE ] && { echo "day17 already done, exits $(date)" >> $L; exit 0; }
exec 9>/tmp/day17.lock
flock -n 9 || { echo "day17 already running, exits $(date)" >> $L; exit 0; }
echo "day17 starts $(date)" >> $L
until [ -f temp/e333/E333_DONE ]; do sleep 60; done
V=$(grep "^verdict:" temp/e333/rule.log 2>/dev/null | tail -1)
case "$V" in
  "verdict: ADOPT"*) EXP=e340 ;;
  *) EXP=e339 ;;
esac
echo "E333 verdict line: '$V' -> $EXP $(date)" >> $L
until [ -f temp/$EXP/READY ] && [ -f temp/$EXP/ACCEPTED ]; do sleep 60; done
bash temp/$EXP/chain.sh >> $LOGD/${EXP}_chain.out 2>&1; echo "$EXP chain exit $? $(date)" >> $L
touch temp/day17/DAY17_DONE
echo "done $(date)" >> $L
