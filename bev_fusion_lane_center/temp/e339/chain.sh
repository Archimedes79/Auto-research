#!/bin/bash
# E339 chain (width 40): both seeds: maps (taught by E307) -> maps score -> kill check -> pairs -> post-processing. usage: chain.sh [marker file to wait for]   log temp/e339/chain.log, marker temp/e339/E339_DONE
cd "$(dirname "$0")/../.."
mkdir -p temp/e339
L=temp/e339/chain.log; LOGD=temp/e339
. temp/e330/chain_lib.sh
wait_marker "$1"
lane e339 k 0 e339_m_s0 e339_p_s0 &
lane e339 k 1 e339_m_s1 e339_p_s1 &
wait
bash tools/env/_py14.sh jrule_removal.py --arm e339_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule.log 2>&1
touch temp/e339/E339_DONE
echo "done $(date)" >> $L
