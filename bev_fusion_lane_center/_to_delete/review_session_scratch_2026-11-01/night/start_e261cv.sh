cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
python3 ../autoresearch/tools/run.py --id E261.cv --require-card --split val --basis E219 \
  --result experiments/E261_recall-weight-6-operating-point/e261_cv.json -- bash temp/loop22/run_E261cv.sh > temp/loop22/E261cv.launcher.log 2>&1
echo "E261.cv: $(tail -n 1 temp/loop22/E261cv.launcher.log)"
