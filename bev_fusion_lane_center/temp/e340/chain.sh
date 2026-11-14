#!/bin/bash
# E340 chain (runs only when E333 is adopted, by temp/day17/day17.sh): both seeds: maps (width 32) -> maps score -> kill check -> pairs -> post-processing. usage: chain.sh [marker file to wait for]   log temp/e340/chain.log, marker temp/e340/E340_DONE
cd "$(dirname "$0")/../.."
mkdir -p temp/e340
L=temp/e340/chain.log; LOGD=temp/e340
. temp/e330/chain_lib.sh
wait_marker "$1"
lane e340 m 0 e340_m_s0 e340_p_s0 &
lane e340 m 1 e340_m_s1 e340_p_s1 &
wait
bash tools/env/_py14.sh jrule_removal.py --arm e340_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule.log 2>&1
touch temp/e340/E340_DONE
echo "done $(date)" >> $L
