#!/bin/bash
# Day chain 13 (2026-11-12 23:55, lane lead): the simplification arms E331 - E334 (D84 point 4) AFTER E320 (D85). Waits for DAY12C_DONE (E320 run and its read-outs done) AND for temp/day13/GO, which the lead writes after reading the E320 result and re-basing (temp/e330/recipe.sh + chain_lib.sh per its header, only if E320 is KEEP).
# Order (value per hour, the window rule of day12: starts 22:00-05:00 and Mon-Fri 08:30-15:30): E331 (evidence off, 1.65 h), E332 arm a (no paint branch, 2.55 h; arm b only if a is adopted, by its own chain), E333 (slim maps, 2.4 h), E334 o3 (one joint stage, 1.65 h). E335 is closed by data.
# Each arm ends with evaluation/jrule_removal.py (exit 0 = ADOPT). Never edit this script while it runs. Log temp/day13/day13.log.
cd "$(dirname "$0")/../.."
L=temp/day13/day13.log; LOGD=temp/day13
mkdir -p $LOGD
echo "day13 waits for DAY12C_DONE and GO $(date)" >> $L
until [ -f temp/day12/DAY12C_DONE ] && [ -f temp/day13/GO ]; do sleep 30; done
echo "go $(date)" >> $L
. temp/e330/chain_lib.sh
# a pair-stage arm, both seeds in parallel: pair_lib <exp lower e.g. e331> <run prefix e.g. e331_e_s>
pair_lib() {
  EXP=$1; PFX=$2
  for S in 0 1; do
    ( wait_start; bash temp/$EXP/start_pairs.sh $S >> $LOGD/${EXP}_lane$S.log 2>&1; echo "$EXP seed $S exit $? $(date)" >> $L; post_pair ${PFX}$S $S ) &
  done
  wait
  bash tools/env/_py14.sh jrule_removal.py --arm $PFX --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_$EXP.log 2>&1; echo "$EXP rule exit $? $(date)" >> $L
}
pair_lib e331 e331_e_s
bash temp/e332/chain.sh >> $LOGD/e332_chain.out 2>&1; echo "e332 chain exit $? $(date)" >> $L
bash temp/e333/chain.sh >> $LOGD/e333_chain.out 2>&1; echo "e333 chain exit $? $(date)" >> $L
pair_lib e334 e334_o_s
touch temp/day13/DAY13_DONE
echo "day13 done $(date)" >> $L
