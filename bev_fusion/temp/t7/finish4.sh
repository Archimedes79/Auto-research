#!/usr/bin/env bash
# t7 finish, part 4 (one-off): finish3.sh with the free-GPU threshold at 2 GB (finish3 waited for < 1 GB, but the Windows desktop apps hold 1.3 GB with no training running; stopped 23:30 on 2026-10-31). Part 3 was: the evaluation half of finish2.sh, which was stopped at 21:11 on 2026-10-31 because its first
# evaluation started together with the t10.dual_s0 training (both waited for the same free GPU; 11.5 GB in use, over the 11 GB rule).
# 23:55: run by the overnight lead's chain (D28) after t10.dual_s1. The skip test was "a manifest exists", but the store writes its
# manifest when an evaluation STARTS, so t7_res4_s0 (evaluation killed at 21:11: manifest, no metrics) would have been skipped; an
# arm is now done only with a metrics.json of GT 15efbde2d1, else its eval/ is moved aside and evaluated afresh.
# T7_NO_WAIT=1 skips the idle wait (room: 5 minutes with < 2 GB GPU, > 10 GB RAM free, no training / evaluation) when the caller
# already holds a GPU slot, e.g. under run.py --gpu; the timing step still runs only on an idle GPU.
# Waits until t10's training has ended and the GPU is free, then
#  1. re-evaluates the four t7 arms with the current suite (same GT signature 15efbde2d1 as t8_ctl_s0; the first evaluations are kept
#     as eval_2026-10-31_first/, the partial one of 21:09 as eval_partial_2026-10-31/),
#  2. paired comparisons: each arm against t2_ref_s<seed> and against t8_ctl_s<seed>, and t8_ctl_s<seed> against t2_ref_s<seed>,
#  3. times the chain on an idle GPU (< 10 % utilisation, < 2 GB used, no training or evaluation), otherwise skipped.
# Log: temp/t7/finish4.log
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
sleep 36000 & SLEEP=$!                                            # keeps WSL alive while this runs
trap 'kill $SLEEP 2> /dev/null' EXIT
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
ram_free() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }   # GB
others() { pgrep -f "[c]ode/train.py|[e]valuation/bev_suite.py" | wc -l; }      # trainings / evaluations of any chat
room() {  # five checks a minute apart: < 2 GB GPU used (the desktop apps hold about 1.3 GB), > 10 GB RAM available, no training or evaluation process
  local ok=0
  while [ $ok -lt 5 ]; do
    if [ "$(gpu_used)" -lt 2000 ] && [ "$(ram_free)" -gt 10 ] && [ "$(others)" -eq 0 ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
}
until ! pgrep -f "[r]uns/t10_dual_s0/code/train.py" > /dev/null; do sleep 300; done
echo "t10 training over $(date +%T)"
if [ "${T7_NO_WAIT:-0}" = 1 ]; then echo "no idle wait (T7_NO_WAIT=1: the caller holds a GPU slot)"; else room; fi
echo "GPU free $(date +%T)"
for a in t7_res4_s0 t7_rgb_s0 t7_res4_s1 t7_rgb_s1; do
  r=runs/$a
  [ -f $r/.done ] || { echo "$a: no .done"; continue; }
  if ! grep -q '"gt_signature": "0.5_4_15efbde2d1"' $r/eval/metrics.json 2> /dev/null; then   # not finished on this GT
    if [ -d $r/eval ] && [ ! -d $r/eval_2026-10-31_first ]; then mv $r/eval $r/eval_2026-10-31_first
    elif [ -d $r/eval ]; then mv $r/eval $r/eval_partial_2026-10-31; fi
    $PY -u evaluation/bev_suite.py $r --workers 2 eval --n 1000 > temp/t7/eval2_$a.log 2>&1
  fi
  echo "$a: $(tail -n 1 temp/t7/eval2_$a.log)"
done
for s in 0 1; do
  for b in t7_res4_s$s t7_rgb_s$s t8_ctl_s$s; do
    [ -f runs/$b/eval/metrics.json ] || continue
    $PY -u evaluation/bev_suite.py compare runs/t2_ref_s$s runs/$b --n 1000 > temp/t7/compare2_ref_$b.txt 2>&1
  done
  [ -f runs/t8_ctl_s$s/eval/metrics.json ] || continue
  for b in t7_res4_s$s t7_rgb_s$s; do
    [ -f runs/$b/eval/metrics.json ] || continue
    $PY -u evaluation/bev_suite.py compare runs/t8_ctl_s$s runs/$b --n 1000 > temp/t7/compare2_ctl_$b.txt 2>&1
  done
done
# t8_ctl_s1 crashed at step 2500 (CUDA error) and has no evaluation: the seed-1 arms against the seed-0 control instead
# (same GT and frames, not seed-paired: the interval then carries the seed spread too)
if [ ! -f runs/t8_ctl_s1/eval/metrics.json ] && [ -f runs/t8_ctl_s0/eval/metrics.json ]; then
  for b in t7_res4_s1 t7_rgb_s1; do
    [ -f runs/$b/eval/metrics.json ] || continue
    $PY -u evaluation/bev_suite.py compare runs/t8_ctl_s0 runs/$b --n 1000 > temp/t7/compare2_ctl0_$b.txt 2>&1
  done
fi
ls temp/t7/compare2_*.txt
if [ "$(gpu_util)" -lt 10 ] && [ "$(gpu_used)" -lt 2000 ] && [ "$(others)" -eq 0 ]; then
  for r in t2_ref_s0 t7_res4_s0 t7_rgb_s0; do
    d=""; [ $r = t2_ref_s0 ] && d="--ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_ref_s0"
    $PY -u evaluation/bev_suite.py runs/$r $d time > temp/t7/time2_$r.log 2>&1
    tail -n 2 temp/t7/time2_$r.log
  done
else
  echo "timing skipped: GPU not idle ($(gpu_util) %, $(gpu_used) MiB, $(others) other jobs)"
fi
echo "t7 finish4 done $(date +%T)"
