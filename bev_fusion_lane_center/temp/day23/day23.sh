#!/bin/bash
# Day chain 23 (2026-11-15, lane experiments chat): the post-processing of E318.u_s1, replacing the part of `temp/day22/day22.sh` that waited for it.
# Why this file exists: the lane lead's day22.sh (E318.u_s1 resumed after an ENOMEM crash at step 69, 01:10) was OVERWRITTEN by the lane experiments chat at 01:23 (a file-name collision: both chats took temp/day22/day22.sh); a running bash
# reads a script by offset, so the lead's process (pid 434595) was killed at 01:3x, its children (temp/e318/start_pairs.sh 1 and the launcher run.py --id E318.u_s1) were left running, and this chain takes over: it waits for the launcher
# of E318.u_s1 to end, then runs what `e318_seed` of day20.sh runs after the training (post_pair, sigma_check, the reads of the card E318), and writes the marker temp/day22/DAY22_DONE that the lane lead expected.
# The recipe of E318.u_s1 is unchanged (workers 3, as its control E317.g). Never edit while it runs. Log temp/day23/day23.log.
cd "$(dirname "$0")/../.."
D=temp/day23
mkdir -p $D
L23=$D/day23.log
exec 9>/tmp/day23.lock
flock -n 9 || { echo "day23 already runs $(date)" >> $L23; exit 0; }
[ -f $D/DAY23_DONE ] && { echo "day23 already done $(date)" >> $L23; exit 0; }
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
T318=temp/e318
echo "day23 starts, waits for the launcher of E318.u_s1 $(date)" >> $L23
while pgrep -f "[r]un.py --id E318.u_s1 " > /dev/null; do sleep 60; done
echo "launcher of E318.u_s1 ended $(date)" >> $L23
S=1; R=e318_u_s1; LA=$PWD/$T318
L=$T318/chain.log; LOGD=$T318; CONTROL_PREFIX=e317_g_s; READOUT_EXTRA="--snap --snap-nogeom"
[ -f runs/$R/model.pt ] || { echo "e318 seed $S (resumed by the lane lead, day22 / day23): no model, post-processing not run $(date)" >> $L; echo "no model of $R $(date)" >> $L23; exit 1; }
echo "e318 seed $S (resumed): training finished, post-processing by day23 $(date)" >> $L
post_pair $R $S
[ -f runs/$R/ends_val.pkl ] && bash tools/env/_py14.sh sigma_check.py $R >> $LOGD/sigma_check_s$S.log 2>&1
bash tools/env/_py14.sh jdiag.py kperr $R e317_g_s$S e312_p_s$S >> $LOGD/kperr_s$S.log 2>&1
(cd temp/e330 && bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> $LA/flag_maps_s$S.log 2>&1)             # REPORT-ONLY
for C in snap snap_nogeom; do
  for B in e317_g_s e312_p_s e324_s_s; do
    [ -f runs/${B}$S/ends_card_${C}_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_${C}_thr0.1 ${B}$S ends_card_${C}_thr0.1 >> $LOGD/boot_${C}_vs_${B}${S}.log 2>&1
  done
done
[ -f runs/e317_g_s$S/ends_card_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e317_g_s$S ends_card_thr0.1 >> $LOGD/boot_unsnap_vs_e317_g_s$S.log 2>&1
echo "e318 seed $S (resumed): post-processing done $(date)" >> $L
touch temp/day22/DAY22_DONE $D/DAY23_DONE
echo "done $(date)" >> $L23
