#!/usr/bin/env bash
# t7 finish, part 2 (one-off): after the GPU lane 4 (t8 + the same-code controls t8_ctl_s0/s1) has ended
#  1. resumes t7_rgb_s1 (died at step ~1000 of an out-of-memory error on the shared box; checkpoint 500) at the pinned 4719d1d,
#  2. re-evaluates the four t7 arms with the current suite, so arms, references and t8_ctl are all scored on the same GT
#     (the GT is built by the live autogt/; the first evaluations of res4_s0 / rgb_s0 ran on an older generator,
#     the old eval folders are kept as eval_2026-10-31_first/),
#  3. paired comparisons: each arm seed against t2_ref_s<seed> and against t8_ctl_s<seed>, and t8_ctl_s<seed> against
#     t2_ref_s<seed> (what the GT change alone does),
#  4. times the chain on an idle GPU (< 10 % utilisation, < 1 GB used), otherwise skipped.
# Log: temp/t7/finish2.log
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
sleep 36000 &                                                     # keeps WSL alive
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
ram_free() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }   # GB
room() {  # three checks a minute apart with < 3 GB GPU used and > 10 GB RAM available
  local ok=0
  while [ $ok -lt 3 ]; do
    if [ "$(gpu_used)" -lt 3000 ] && [ "$(ram_free)" -gt 10 ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
}
until grep -q "lane4 finished" temp/study/lane4.log 2>/dev/null || ! pgrep -f "bash temp/study/lane4.sh" > /dev/null; do sleep 300; done
echo "lane 4 over $(date +%T); autogt at $(git log -1 --format=%h -- autogt), working tree $(git diff --quiet a49bbd3 -- autogt && echo = a49bbd3 || echo CHANGED since a49bbd3)"
# partial (0-byte) GT cache files of killed writers crash a training: move them aside
mkdir -p _to_delete/gt_cache_partial
find -L data/cache/gt -maxdepth 1 -name "*.npz" -size 0 -print -exec mv {} _to_delete/gt_cache_partial/ \;
room
r=runs/t7_rgb_s1
if [ ! -f $r/.done ]; then
  [ -f $r/train_crash_2026-10-31.log ] || mv $r/train.log $r/train_crash_2026-10-31.log
  echo "== t7_rgb_s1 resume $(date +%T)"
  bash temp/train/run_cfg.sh training/configs/t7_rgb_s1.json 4719d1d
fi
[ -f $r/.done ] || { echo "t7_rgb_s1: no .done"; }
room
for a in t7_res4_s0 t7_rgb_s0 t7_res4_s1 t7_rgb_s1; do
  r=runs/$a
  [ -f $r/.done ] || continue
  [ -d $r/eval ] && [ ! -d $r/eval_2026-10-31_first ] && mv $r/eval $r/eval_2026-10-31_first
  [ -f $r/eval/suite/quant/eval_n1000/manifest.json ] || \
    $PY -u evaluation/bev_suite.py $r --workers 2 eval --n 1000 > temp/t7/eval2_$a.log 2>&1
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
if [ "$(gpu_util)" -lt 10 ] && [ "$(gpu_used)" -lt 1000 ]; then
  for r in t2_ref_s0 t7_res4_s0 t7_rgb_s0; do
    d=""; [ $r = t2_ref_s0 ] && d="--ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_ref_s0"
    $PY -u evaluation/bev_suite.py runs/$r $d time > temp/t7/time2_$r.log 2>&1
    tail -n 2 temp/t7/time2_$r.log
  done
else
  echo "timing skipped: GPU not idle ($(gpu_util) %, $(gpu_used) MiB)"
fi
echo "t7 finish2 done $(date +%T)"
