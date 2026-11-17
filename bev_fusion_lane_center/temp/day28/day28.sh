#!/bin/bash
# day28 = the GPU BRANCH of the E343 dress rehearsal on VAL (card experiments/E343_test-e324s/experiment.md section 2, criterion 2: "If a tolerance is exceeded the GPU branch applies and nothing else"): the fresh CPU rehearsal of
# E312 (temp/day27, step all_e312) exceeded the pre-registered tolerance (vertices per 100 m 3.01e-3 / 8.24e-3 against 3e-3; wired 5996 against 5992 and curves 11842 against 11849, relative 6.7e-4 / 5.9e-4 against 5e-4), so the
# pair head of E324 s (and c2's B for arm k) are dumped on the GPU, the device of the stored VAL dumps; B's E273 dump stays the fresh CPU one (E321's on TEST) for all arms. The same command as the TEST line with VAL files and --device cuda,
# one run.py diagnostic job (--no-card) that takes a GPU slot (first come first served), the dumps wait for GPU room (evaluation/gpu_gate.py), 10 GB MemAvailable before the start, the peak RSS of the process is sampled.
# Marker temp/day28/DAY28_DONE; log temp/day28/day28.log. Nothing under evaluation/ or training/ may change while it runs (the code digest of the rehearsal is the one the TEST line must carry).
# usage: wsl bash temp/day28/day28.sh   (DAY28_DRY=1: only log the command)
cd "$(dirname "$0")/../.."
D=temp/day28; L=$D/day28.log; mkdir -p $D
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
say() { echo "$* $(date)" >> $L; }
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
. temp/e330/chain_lib.sh
quiet() {
  local t=0
  while [ "$(avail_mb)" -lt 10000 ]; do
    sleep 60; t=$((t + 1)); [ $t -ge 720 ] && { say "quiet: gave up after 12 h (MemAvailable $(avail_mb) MB)"; return 1; }
  done
  say "start allowed: MemAvailable $(avail_mb) MB"
}
sampler() {
  local out=$1 h p
  while true; do
    for p in $(pgrep -f "python jtest_eval.py"); do
      h=$(awk '/VmHWM/ {print int($2/1024)}' /proc/$p/status 2>/dev/null); [ -n "$h" ] && echo "$h" >> $out
    done
    sleep 20
  done
}
for f in temp/e343/val/predfull_e273_p_s0.pkl temp/e343/val/predfull_e273_p_s1.pkl temp/e343/val/ends_e312_p_s0.pkl temp/e343/val/ends_e312_p_s1.pkl runs/eval/e312_val/card_s0.json runs/eval/e312_val/card_s1.json; do
  [ -f $f ] || { say "missing input $f"; exit 1; }
done
[ -d runs/eval/e324s_val ] && { say "runs/eval/e324s_val exists: move it away first (cards are written once)"; exit 1; }
say "day28 starts (dry: ${DAY28_DRY:-no})"
CMD="bash tools/env/_py14.sh jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cuda --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json"
if [ -n "$DAY28_DRY" ]; then say "DRY run.py --id E343.rh_gpu_dumps --gpu --no-card -- bash temp/day28/gpu_dumps.sh"; say "DRY run.py --id E343.rh_all_e324s_gpu --no-card -- $CMD"; exit 0; fi
# stage 1 (GPU slot, about 25 minutes): the four GPU dumps;  stage 2 (CPU): `all` reuses them under the same provenance (device cuda) and scores on the CPU
rp() {
  local id=$1 gpu=$2; shift 2
  local fl=""; [ "$gpu" = 1 ] && fl="--gpu"
  quiet || return 1
  sampler $D/rss_samples_$id.txt & local sp=$!
  python3 ../autoresearch/tools/run.py --id $id $fl --retries 0 --no-card --split val --timeout 720     --hyp "E343 criterion 2, GPU branch: the dress rehearsal of jtest_eval.py on VAL with the pair head of E324 s (and c2's B) dumped on the GPU; diagnostic, feeds no verdict; job $id"     --pred "no number is predicted: the rehearsal must run to the end, reproduce the stored VAL cards of E324 s within the pre-registered tolerance and write cards with a code digest, device cuda"     --rule "by hand: a check of the code path, not a comparison; criterion 2 of the E343 card (GPU branch)"     -- "$@"
  local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  [ -s $D/rss_samples_$id.txt ] && say "peak RSS of $id: $(sort -n $D/rss_samples_$id.txt | tail -1) MB"
  return $rc
}
if [ ! -f $D/gpu_dumps.done ]; then
  wait_start
  if rp E343.rh_gpu_dumps 1 bash temp/day28/gpu_dumps.sh; then touch $D/gpu_dumps.done; say "gpu dumps done"; else touch $D/FAIL_gpu_dumps; say "gpu dumps FAILED"; exit 1; fi
fi
if rp E343.rh_all_e324s_gpu 0 $CMD; then touch $D/DAY28_DONE; say "day28 done (exit 0)"; else touch $D/FAIL_all_e324s_gpu; say "day28 FAILED"; exit 1; fi
