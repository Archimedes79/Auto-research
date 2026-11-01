#!/usr/bin/env bash
# times the selftests that tests/run_gates.sh may add (CPU only, 2 threads, nice); run from the bev_fusion folder
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=/home/ml/miniconda3/envs/gcar/bin/python
OUT=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/coderC/selftests
mkdir -p $OUT
for s in tests/test_gt_cache_targets.py training/augment.py training/nuscenes_dataset.py training/model.py training/train.py training/losses.py \
         training/legacy/s1_camera.py training/legacy/camera_decoder.py training/legacy/radar_decoder.py training/legacy/light_encoder.py training/legacy/dualbev_front.py; do
  n=$(echo $s | tr '/' '_')
  t0=$(date +%s)
  nice -n 10 $PY $s > $OUT/$n.log 2>&1
  rc=$?
  echo "$s exit $rc $(( $(date +%s) - t0 )) s | $(grep -v -i warn $OUT/$n.log | tail -n 1 | cut -c1-160)"
done
echo "all timed"
