#!/bin/bash
# scratch: the GPU window of the big report (Mirko 2026-11-02): overview videos of random validation scenes (seed 21, the same scenes for every run), the lane-divider gallery frames
# (worst / median / best of different scenes, temp/t16/lane_frames_t16_full_s0.txt) and the like-for-like re-scorings against the state of the art
# (temp/t16/soa_maps_cmd.sh, soa_front_cmd.sh when they exist). It runs as the GPU job showcase.gpu behind t18.fix; the control trainer waits for it.
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=/home/ml/miniconda3/envs/gcar/bin/python
S=evaluation/bev_suite.py
for RUN in t16_full_s0 t17_data; do
  N=6
  [ "$RUN" = "t16_full_s0" ] && N=8
  echo "== videos $RUN $(date +%T)"
  $PY $S runs/$RUN --out runs/$RUN/eval/showcase qual videos $N --view overview --seed 21 2>&1 | tail -3 | cut -c1-160 || echo "videos $RUN failed"
done
echo "== gallery $(date +%T)"
for lab in $(grep "^PICK" temp/t16/lane_frames_t16_full_s0.txt | awk '{print $3}'); do
  scene=${lab%/*}; idx=$((10#${lab#*/}))
  $PY $S runs/t16_full_s0 --out runs/t16_full_s0/eval/showcase qual frame $scene/$idx --view overview --history warmup 2>&1 | tail -1 | cut -c1-160 || echo "frame $lab failed"
done
if [ -f temp/t16/soa_maps_cmd.sh ]; then echo "== soa maps $(date +%T)"; bash temp/t16/soa_maps_cmd.sh || echo "soa maps failed"; fi
if [ -f temp/t16/soa_front_cmd.sh ]; then echo "== soa front $(date +%T)"; bash temp/t16/soa_front_cmd.sh || echo "soa front failed"; fi
echo "== done $(date +%T)"
