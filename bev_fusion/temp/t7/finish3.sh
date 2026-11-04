#!/usr/bin/env bash
# t7 finish, part 3 (one-off): the evaluation half of finish2.sh, which was stopped at 21:11 on 2026-10-31 because its first
# evaluation started together with the t10.dual_s0 training (both waited for the same free GPU; 11.5 GB in use, over the 11 GB rule).
# Waits until t10's training has ended and the GPU is free, then
#  1. re-evaluates the four t7 arms with the current suite (same GT signature 15efbde2d1 as t8_ctl_s0; the first evaluations are kept
#     as eval_2026-10-31_first/, the partial one of 21:09 as eval_partial_2026-10-31/),
#  2. paired comparisons: each arm against t2_ref_s<seed> and against t8_ctl_s<seed>, and t8_ctl_s<seed> against t2_ref_s<seed>,
#  3. times the chain on an idle GPU (< 10 % utilisation, < 1 GB used), otherwise skipped.
# Log: temp/t7/finish3.log
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
sleep 36000 &                                                     # keeps WSL alive
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
ram_free() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }   # GB
others() { pgrep -f "[c]ode/train.py|[e]valuation/bev_suite.py" | wc -l; }      # trainings / evaluations of any chat
room() {  # five checks a minute apart: < 1 GB GPU used, > 10 GB RAM available, no training or evaluation process
  local ok=0
  while [ $ok -lt 5 ]; do
    if [ "$(gpu_used)" -lt 1000 ] && [ "$(ram_free)" -gt 10 ] && [ "$(others)" -eq 0 ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
}
until ! pgrep -f "[r]uns/t10_dual_s0/code/train.py" > /dev/null; do sleep 300; done
echo "t10 training over $(date +%T)"
room
echo "GPU free $(date +%T)"
for a in t7_res4_s0 t7_rgb_s0 t7_res4_s1 t7_rgb_s1; do
  r=runs/$a
  [ -f $r/.done ] || { echo "$a: no .done"; continue; }
  if [ ! -f $r/eval/suite/quant/eval_n1000/manifest.json ]; then
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
ls temp/t7/compare2_*.txt
if [ "$(gpu_util)" -lt 10 ] && [ "$(gpu_used)" -lt 1000 ] && [ "$(others)" -eq 0 ]; then
  for r in t2_ref_s0 t7_res4_s0 t7_rgb_s0; do
    d=""; [ $r = t2_ref_s0 ] && d="--ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_ref_s0"
    $PY -u evaluation/bev_suite.py runs/$r $d time > temp/t7/time2_$r.log 2>&1
    tail -n 2 temp/t7/time2_$r.log
  done
else
  echo "timing skipped: GPU not idle ($(gpu_util) %, $(gpu_used) MiB, $(others) other jobs)"
fi
echo "t7 finish3 done $(date +%T)"
