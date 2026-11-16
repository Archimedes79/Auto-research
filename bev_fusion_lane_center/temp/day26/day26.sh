#!/bin/bash
# day26 = E343 criterion 2, the dress rehearsal on VAL (card experiments/E343_test-e324s/experiment.md; sequence (b) of the docstring of evaluation/jtest_eval.py), CPU only, one heavy CPU job at a time,
# each step a run.py diagnostic job (--no-card: it never feeds a verdict; the ledger row says so). Markers temp/day26/<step>.done, a failure stops the chain (temp/day26/FAIL_<step>), log temp/day26/day26.log.
# usage: wsl bash temp/day26/day26.sh   (DAY26_DRY=1: only log the commands).  Nothing under evaluation/ or training/ may change while it runs (the code digest of the rehearsal is the one the TEST line must carry).
cd "$(dirname "$0")/../.."
D=temp/day26; L=$D/day26.log; mkdir -p $D
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say() { echo "$* $(date)" >> $L; }
# quiet: wait (up to 12 h) until 8 GB of RAM are free (the lane lead put E343 first, 2026-11-15 03:20: the E345 scoring jobs are not waited for; one heavy job of this chain at a time)
quiet() {
  local t=0
  while [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 8000 ]; do
    sleep 60; t=$((t + 1)); [ $t -ge 720 ] && { say "quiet: gave up after 12 h"; return 1; }
  done
}
# rp <run id> <command ...>: one run.py diagnostic CPU job (blocks until it ends)
rp() {
  local id=$1; shift
  if [ -n "$DAY26_DRY" ]; then echo "DRY run.py --id $id --no-card -- $*" >> $L; return 0; fi
  quiet || return 1
  python3 ../autoresearch/tools/run.py --id $id --retries 0 --no-card --split val --timeout 720 \
    --hyp "E343 criterion 2: the dress rehearsal of jtest_eval.py on VAL (the TEST command with VAL files); diagnostic, feeds no verdict; job $id" \
    --pred "no number is predicted: the rehearsal must run to the end, reproduce the E312 cards (plain graph within 3e-3 for a fresh CPU dump) and write cards with a code digest" \
    --rule "by hand: a check of the code path, not a comparison; criterion 2 of the E343 card" \
    -- "$@"
}
stage() { local n=$1; shift; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if "$@"; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
J="bash tools/env/_py14.sh jtest_eval.py"
say "day26 starts (dry: ${DAY26_DRY:-no})"
for S in 0 1; do
  stage dumpb_$S    rp E343.rh_dumpb_s$S    $J dump-b    --split val --seed $S --device cpu
  stage dumpends_$S rp E343.rh_dumpends_s$S $J dump-ends --split val --seed $S --run e312_p_s$S --device cpu
done
stage all_e312 rp E343.rh_all_e312 $J all --split val --run-prefix e312_p_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl
stage all_e324s rp E343.rh_all_e324s $J all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl \
  --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json
touch $D/DAY26_DONE; say "day26 done"
