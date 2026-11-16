#!/bin/bash
# Day chain 19 (2026-11-14, lane experiments chat, D96 / D98): E317 -> E318 (arm u, control E317.g) -> E319, ONE GPU job at a time, after the chain of E341 has finished (day18). Each chain waits for the window and for host_ok itself.
# Starts only after the holds of E317.g_s*, E318.u_s*, E319.m_s*, E319.p_s* are cleared and temp/day19/GO exists (written after the critic's second pass). Run-once guard. Never edit this script while it runs.
# Log temp/day19/day19.log, marker temp/day19/DAY19_DONE.
cd "$(dirname "$0")/../.."
L=temp/day19/day19.log
mkdir -p temp/day19
exec 9>/tmp/day19.lock
flock -n 9 || { echo "day19 already runs $(date)" >> $L; exit 0; }
[ -f temp/day19/DAY19_DONE ] && { echo "day19 already done $(date)" >> $L; exit 0; }
echo "day19 starts $(date)" >> $L
until [ -f temp/day19/GO ]; do sleep 60; done
until [ -f temp/day18/DAY18_DONE ] || grep -q "e341 chain exit" temp/day18/day18.log 2>/dev/null; do sleep 120; done
echo "e341 chain finished, E317 starts $(date)" >> $L
bash temp/e317/chain.sh >> temp/day19/e317_chain.out 2>&1; echo "e317 chain exit $? $(date)" >> $L
bash temp/e318/chain.sh >> temp/day19/e318_chain.out 2>&1; echo "e318 chain exit $? $(date)" >> $L
bash temp/e319/chain.sh >> temp/day19/e319_chain.out 2>&1; echo "e319 chain exit $? $(date)" >> $L
touch temp/day19/DAY19_DONE
echo "done $(date)" >> $L
