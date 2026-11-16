#!/bin/bash
# day27 = the rest of the E343 dress rehearsal on VAL (card experiments/E343_test-e324s/experiment.md; sequence (b) of the docstring of evaluation/jtest_eval.py), handed over from temp/day26/day26.sh at the lane lead's request (2026-11-15 05:25):
# every step starts only when MemAvailable >= 10 GB (day26 had 8 GB and a running script cannot be edited: its remaining stages were skipped by markers and this chain takes over after day26 ended), ONE heavy job at a time,
# the peak resident memory of every step is sampled (temp/day27/rss_<step>.txt: the highest VmHWM seen of the jtest_eval.py process, MB). Each step is a run.py diagnostic job (--no-card), markers temp/day27/<step>.done,
# a failure stops the chain (temp/day27/FAIL_<step>), log temp/day27/day27.log.  Nothing under evaluation/ or training/ may change while it runs (the code digest of the rehearsal is the one the TEST line must carry).
# usage: wsl bash temp/day27/day27.sh   (DAY27_DRY=1: only log the commands)
cd "$(dirname "$0")/../.."
D=temp/day27; L=$D/day27.log; mkdir -p $D
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say() { echo "$* $(date)" >> $L; }
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
# wait (up to 12 h) until day26 ended (its last step is dumpb_1) and its outputs are there
if [ -z "$DAY27_DRY" ]; then
  t=0
  until [ -f temp/day26/DAY26_DONE ] && [ -f temp/day26/dumpb_1.done ] && [ -f temp/e343/val/predfull_e273_p_s1.pkl ]; do
    sleep 30; t=$((t + 1)); [ $t -ge 1440 ] && { say "gave up waiting for day26"; exit 1; }
  done
fi
# quiet: wait (up to 12 h) until 10 GB of RAM are available
quiet() {
  local t=0
  while [ "$(avail_mb)" -lt 10000 ]; do
    sleep 60; t=$((t + 1)); [ $t -ge 720 ] && { say "quiet: gave up after 12 h (MemAvailable $(avail_mb) MB)"; return 1; }
  done
  say "start allowed: MemAvailable $(avail_mb) MB"
}
# sampler <file>: every 20 s the highest VmHWM (MB) of a running jtest_eval.py process -> the file (the peak is its maximum)
sampler() {
  local out=$1 h p
  while true; do
    for p in $(pgrep -f "python jtest_eval.py"); do
      h=$(awk '/VmHWM/ {print int($2/1024)}' /proc/$p/status 2>/dev/null); [ -n "$h" ] && echo "$h" >> $out
    done
    sleep 20
  done
}
# rp <run id> <command ...>: one run.py diagnostic CPU job (blocks until it ends)
rp() {
  local id=$1; shift
  if [ -n "$DAY27_DRY" ]; then echo "DRY run.py --id $id --no-card -- $*" >> $L; return 0; fi
  quiet || return 1
  sampler $D/rss_samples_$id.txt & local sp=$!
  python3 ../autoresearch/tools/run.py --id $id --retries 0 --no-card --split val --timeout 720 \
    --hyp "E343 criterion 2: the dress rehearsal of jtest_eval.py on VAL (the TEST command with VAL files); diagnostic, feeds no verdict; job $id" \
    --pred "no number is predicted: the rehearsal must run to the end, reproduce the E312 cards (plain graph within 3e-3 for a fresh CPU dump) and write cards with a code digest" \
    --rule "by hand: a check of the code path, not a comparison; criterion 2 of the E343 card" \
    -- "$@"
  local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  [ -s $D/rss_samples_$id.txt ] && say "peak RSS of $id: $(sort -n $D/rss_samples_$id.txt | tail -1) MB"
  return $rc
}
stage() { local n=$1; shift; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if "$@"; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
J="bash tools/env/_py14.sh jtest_eval.py"
say "day27 starts (dry: ${DAY27_DRY:-no})"
stage dumpends_1 rp E343.rh_dumpends_s1 $J dump-ends --split val --seed 1 --run e312_p_s1 --device cpu
stage all_e312 rp E343.rh_all_e312 $J all --split val --run-prefix e312_p_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl
stage all_e324s rp E343.rh_all_e324s $J all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl \
  --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json
touch $D/DAY27_DONE; say "day27 done"
