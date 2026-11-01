cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
sleep 90   # no simultaneous run.py start with start_e264b.sh
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
python3 ../autoresearch/tools/run.py --id E266.c --require-card --split val \
  -- $T bash -c 'bash tools/env/_py14.sh evaluate.py recall --tag l22r --expect Q=0.7404 lanes_found=0.7673 R@1_far=0.7714 && bash tools/env/_py14.sh evaluate.py recall_s1 --tag l22r --expect Q=0.7376 lanes_found=0.7695 R@1_far=0.7707 && bash tools/env/_py14.sh evaluate.py fast_recall --tag l22r --expect Q=0.7392 lanes_found=0.7364 R@1_far=0.7275 && bash tools/env/_py14.sh evaluate.py e24_j1rc_s1 --tag l22r --expect Q=0.7427 lanes_found=0.7433 R@1_far=0.7368' > temp/loop22/E266c.launcher.log 2>&1
echo "E266.c: $(tail -n 1 temp/loop22/E266c.launcher.log)"
python3 ../autoresearch/tools/run.py --id E267.s --require-card --split val \
  -- $T bash -c 'bash tools/env/_py14.sh evaluate.py recall --asm spur_len=0 --tag sp0r --compare recall && bash tools/env/_py14.sh evaluate.py recall_s1 --asm spur_len=0 --tag sp0r --compare recall_s1' > temp/loop22/E267s.launcher.log 2>&1
echo "E267.s: $(tail -n 1 temp/loop22/E267s.launcher.log)"
