#!/bin/bash
# Day chain 18 (2026-11-14, lane chat, D92): E341 = the evidence along the curve without the 0.5 m paint maps on the maps head without the 0.5 m branch (the width stays 64). Starts nothing before temp/e341/READY and temp/e341/ACCEPTED
# (the critic pass) exist, only in the GPU windows (chain_lib.sh: nights 22:00 - 05:00, Mon - Fri 08:30 - 15:30, and since D98 Saturdays and Sundays all day), once. Never edit this script while it runs.  Log temp/day18/day18.log, marker temp/day18/DAY18_DONE.
cd "$(dirname "$0")/../.."
L=temp/day18/day18.log
mkdir -p temp/day18
exec 9>/tmp/day18.lock
flock -n 9 || { echo "day18 already runs $(date)" >> $L; exit 0; }
[ -f temp/day18/DAY18_DONE ] && { echo "day18 already done $(date)" >> $L; exit 0; }
echo "day18 starts $(date)" >> $L
until [ -f temp/e341/READY ] && [ -f temp/e341/ACCEPTED ]; do sleep 60; done
echo "e341 READY and ACCEPTED $(date)" >> $L
bash temp/e341/chain.sh >> temp/day18/e341_chain.out 2>&1; RC=$?; echo "e341 chain exit $RC $(date)" >> $L
[ $RC = 0 ] && touch temp/day18/DAY18_DONE                                                              # a failed chain leaves no done marker: a re-run needs only temp/e341/E341_DONE removed
echo "done $(date)" >> $L
