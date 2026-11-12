#!/bin/bash
# E333 chain (NOT started; scheduled AFTER E320, D85): both seeds: maps (width 40) -> maps score -> kill check -> pairs -> post-processing. usage: chain.sh [marker file to wait for]   log temp/e333/chain.log, marker temp/e333/E333_DONE
cd "$(dirname "$0")/../.."
mkdir -p temp/e333
L=temp/e333/chain.log; LOGD=temp/e333
. temp/e330/chain_lib.sh
wait_marker "$1"
lane e333 m 0 e333_m_s0 e333_p_s0 &
lane e333 m 1 e333_m_s1 e333_p_s1 &
wait
bash tools/env/_py14.sh jrule_removal.py --arm e333_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule.log 2>&1
touch temp/e333/E333_DONE
echo "done $(date)" >> $L
