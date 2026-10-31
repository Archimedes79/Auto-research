#!/usr/bin/env bash
# LANE 4 (step 0 of plan 2.8, feature swap with BEVFusion features): behind lane 3 (waits for "lane finished" in lane3b.log, or starts
# the extraction as soon as the GPU has headroom: used < 7000 MiB on three checks a minute apart).
# 1. extraction of the full BEVFusion feature cache (58,101 frames, 64.8 GB on D: behind the data/cache symlink; resumable per scene)
# 2. t8_bevfeat_s0, t8_bevfeat_s1 (= t2_ref_s0 / s1 with camera_features "bevfusion"), code pinned at commit 96a02be, then the suite evaluation
# Idempotent (skips what exists). Usage: bash temp/study/lane4.sh   (log: temp/study/lane4.log)
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
REV=96a02be
sleep 36000 &
ok=0
while ! grep -q "lane finished" temp/study/lane3b.log; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
  if [ "${used:-99999}" -lt 7000 ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 3 ] && break
  sleep 60
done
echo "== extraction $(date +%T)"
OMP_NUM_THREADS=2 $PY -u training/cache_bevfusion_features.py build --src data/cache/cam_feats --out data/cache/cam_feats_bevfusion --device cuda \
  >> temp/study/bevfeat_extract.log 2>&1
rc=$?
echo "extraction exit $rc $(date +%T)"
[ $rc -eq 0 ] || { echo "extraction failed, no training"; exit 1; }
for s in 0 1; do
  name=t8_bevfeat_s$s
  r=runs/$name
  [ -f $r/eval/metrics.json ] && { echo "$name already evaluated, skip"; continue; }
  echo "== $name train $(date +%T)"
  [ -f $r/.done ] || bash temp/train/run_cfg.sh training/configs/$name.json $REV
  [ -f $r/.done ] || { echo "$name: no .done, skipped evaluation"; continue; }
  echo "== $name eval $(date +%T)"
  $PY -u evaluation/bev_suite.py $r --workers 4 eval --n 1000 > temp/study/eval_$name.log 2>&1
  tail -n 1 temp/study/eval_$name.log
done
echo "lane4 finished $(date +%T)"
