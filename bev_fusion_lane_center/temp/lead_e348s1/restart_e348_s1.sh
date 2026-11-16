#!/bin/bash
# Restart of ONLY the stage E348 seed 1 (lane lead, 2026-11-15 05:3x). E348.w_s1 FAILED at step 5038 of 6000 (04:54) with OSError [Errno 12] "Cannot allocate memory" in a DATA WORKER (the same cause as E318.u_s1 at 01:10:
# memory spike with several chats' jobs at once: t26 base_eval ~9 GB, the E343 rehearsal dumps, E345 scoring, two trainings). ckpt.pt (step 5000) exists: the same id is resumed by run.py (previous log kept) and the
# training resumes from the checkpoint (jends_train.py: "resumed from ... ckpt.pt at step N"); recipe, seed and workers (4, D101) are those of temp/e348/start_pairs.sh, UNCHANGED.
# Post-processing = the lane body of temp/day24/day24.sh for seed 1 (copied; day24 is not edited and goes on with lane 0). day24's final jrule_removal step runs when lane 0 ends, with seed 1 missing (exit 2): this script
# waits for lane 0's "post-processing done" and then runs the rule again (the later line in temp/e348/chain.log counts). D100: no host gate; one wait: MemAvailable >= 8 GB. Never edit while it runs.
# Name outside the dayNN sequence on purpose (a name collision overwrote day22 once). Log temp/lead_e348s1/restart.log and temp/e348/chain.log.
cd "$(dirname "$0")/../.."
D=temp/lead_e348s1; T=temp/e348; mkdir -p $D
exec 9>/tmp/lead_e348s1.lock
flock -n 9 || { echo "already runs $(date)" >> $D/restart.log; exit 0; }
[ -f $D/DONE ] && { echo "already done $(date)" >> $D/restart.log; exit 0; }
echo "starts $(date)" >> $D/restart.log
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
X() { "$@"; }; has() { [ -f "$1" ]; }
S=1; R=e348_w_s1; C=e324_s_s1; LA=$PWD/$T
L=$T/chain.log; LOGD=$T; CONTROL_PREFIX=e324_s_s; READOUT_EXTRA="--attach --snap --snap-nogeom"
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 8 ]; do echo "e348 seed 1 restart waits for MemAvailable >= 8 GB $(date)" >> $L; sleep 120; done
wait_start
echo "e348 seed 1 RESTART (lead) of the stage that failed at step 5038 (ENOMEM in a data worker, 04:54), from ckpt step 5000 $(date)" >> $L
for TRY in $(seq 1 30); do
  bash temp/e348/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?
  [ $XC = 2 ] && tail -n 1 $LOGD/lane$S.log | grep -q "still running" || break
  echo "e348 seed 1 restart: the id still counts as running, try $TRY $(date)" >> $L; sleep 60
done
echo "e348 seed $S restart exit $XC $(date)" >> $L
{ [ $XC = 0 ] && has runs/$R/model.pt; } || { echo "e348 seed $S restart FAILED (exit $XC or no model) $(date)" >> $L; exit 1; }
post_pair $R $S
has runs/$R/model.pt || exit 1
X bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --thr 0.1 >> $LOGD/score_attach_$R.log 2>&1
X bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --snap-nogeom --thr 0.1 >> $LOGD/score_nogeom_$R.log 2>&1
X bash tools/env/_py14.sh jdiag.py kperr $R $C >> $LOGD/kperr_s$S.log 2>&1
(cd temp/e330 && X bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> $LA/flag_maps_s$S.log 2>&1)
for B in thr0.1 snap_thr0.1 snap_nogeom_attach_thr0.1 attach_thr0.1 snap_nogeom_thr0.1; do
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_$B $C ends_card_$B >> $LOGD/boot_${B}_s$S.log 2>&1
done
echo "e348 seed $S post-processing done (restart) $(date)" >> $L
until grep -q "e348 seed 0 post-processing done" $L; do echo "waiting for lane 0 of day24 $(date)" >> $D/restart.log; sleep 120; done
X bash tools/env/_py14.sh jrule_removal.py --arm e348_w_s --control e324_s_s --readouts snap primary attach >> $T/rule.log 2>&1
echo "e348 rule exit $? (0 ADOPT, 1 not adopted, 2 a card missing); both seeds after the restart of seed 1 $(date)" >> $L
touch $D/DONE
echo "done $(date)" >> $D/restart.log
