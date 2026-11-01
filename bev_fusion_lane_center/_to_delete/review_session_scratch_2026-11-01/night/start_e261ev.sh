cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
# wait for the other CPU launcher (start_cpu2.sh) to leave run.py: no simultaneous run.py starts
sleep 5
python3 ../autoresearch/tools/run.py --id E261.ev --require-card --split val \
  -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 bash -c 'bash tools/env/_py14.sh evaluate.py recall6 --compare recall && bash tools/env/_py14.sh evaluate.py recall_thr --compare recall' > temp/loop22/E261ev.launcher.log 2>&1
echo "E261.ev: $(tail -n 1 temp/loop22/E261ev.launcher.log)"
