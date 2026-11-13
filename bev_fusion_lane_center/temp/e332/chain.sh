#!/bin/bash
# E332 chain (NOT started; the lane lead schedules it AFTER E320, D85): arm a both seeds (maps -> maps score -> kill check -> pairs -> post-processing), then arm b only when arm a is ADOPTED by jrule_removal.py.
# usage: chain.sh [marker file to wait for]   log temp/e332/chain.log, marker temp/e332/E332_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
mkdir -p temp/e332
L=temp/e332/chain.log; LOGD=temp/e332
. temp/e330/chain_lib.sh
wait_marker "$1"
lane e332 a 0 e332_ma_s0 e332_pa_s0 &
lane e332 a 1 e332_ma_s1 e332_pa_s1 &
wait
if bash tools/env/_py14.sh jrule_removal.py --arm e332_pa_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_a.log 2>&1; then
  echo "arm a ADOPTED: arm b runs $(date)" >> $L
  lane e332 b 0 e332_mb_s0 e332_pb_s0 &
  lane e332 b 1 e332_mb_s1 e332_pb_s1 &
  wait
  bash tools/env/_py14.sh jrule_removal.py --arm e332_pb_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_b.log 2>&1
else
  echo "arm a not adopted (or incomplete): arm b not run $(date)" >> $L
fi
touch temp/e332/E332_DONE
echo "done $(date)" >> $L
