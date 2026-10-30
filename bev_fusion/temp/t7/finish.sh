#!/usr/bin/env bash
# t7 finish (one-off): waits until the four t7 runs of the lane3 queue are evaluated (or failed), then on a GPU with room
# (< 6 GB used) re-evaluates the two references with the current suite (old legacy metrics copied aside first: layer-1 check),
# runs the paired scene bootstrap of each arm seed against the reference seed with the same index, and times the chain
# (only on an idle GPU: < 10 % utilisation and < 1 GB used; otherwise skipped). Log: temp/t7/finish.log
cd "$(dirname "$0")/../.." || exit 1
PY="$HOME/miniconda3/envs/gcar/bin/python"
L=temp/study/lane3b.log                                          # the live log of temp/study/lane3.sh
RUNS="t7_res4_s0 t7_rgb_s0 t7_res4_s1 t7_rgb_s1"
sleep 36000 &                                                     # keeps WSL alive overnight
state() {  # done | failed | wait
  [ -f runs/$1/eval/metrics.json ] && { echo done; return; }
  grep -q "cfg:$1:.*no .done" $L 2>/dev/null && { echo failed; return; }
  echo wait
}
for r in $RUNS; do
  until [ "$(state $r)" != wait ]; do sleep 120; done
  echo "$r: $(state $r) $(date +%T)"
done
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
until [ "$(gpu_used)" -lt 6000 ]; do sleep 120; done
for s in 0 1; do
  r=runs/t2_ref_s$s
  [ -f $r/eval/metrics_legacy_2026-10-28.json ] || cp $r/eval/metrics.json $r/eval/metrics_legacy_2026-10-28.json
  [ -f $r/eval/suite/quant/eval_n1000/manifest.json ] || \
    $PY -u evaluation/bev_suite.py $r --workers 2 --ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_ref_s$s eval --n 1000 \
      > temp/t7/eval_ref_s$s.log 2>&1
  tail -n 1 temp/t7/eval_ref_s$s.log
done
for s in 0 1; do
  for arm in res4 rgb; do
    [ -f runs/t7_${arm}_s$s/eval/metrics.json ] || continue
    $PY -u evaluation/bev_suite.py compare runs/t2_ref_s$s runs/t7_${arm}_s$s --n 1000 > temp/t7/compare_${arm}_s$s.txt 2>&1
    echo "compare ${arm} s$s: $(grep -c excludes temp/t7/compare_${arm}_s$s.txt) metrics with an interval excluding 0"
  done
done
if [ "$(gpu_util)" -lt 10 ] && [ "$(gpu_used)" -lt 1000 ]; then
  for r in t2_ref_s0 t7_res4_s0 t7_rgb_s0; do
    d=""; [ $r = t2_ref_s0 ] && d="--ckpt-dir /mnt/d/Auto-research/bev_fusion_ckpts/t2_ref_s0"
    $PY -u evaluation/bev_suite.py runs/$r $d time > temp/t7/time_$r.log 2>&1
    tail -n 2 temp/t7/time_$r.log
  done
else
  echo "timing skipped: GPU not idle ($(gpu_util) %, $(gpu_used) MiB)"
fi
echo "t7 finish done $(date +%T)"
