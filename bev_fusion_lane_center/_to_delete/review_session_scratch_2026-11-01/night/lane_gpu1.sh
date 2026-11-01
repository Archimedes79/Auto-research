# lane GPU window part 1 (loop 22): E266.b0 (ring raw outputs of recall) on the GPU, then the gates G0 / G1 on the CPU (E266.g01); stops for the lead's gate reading
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
echo 1000 > /proc/self/oom_score_adj
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1; }
# wait until the bev B2 training has ended (its launcher gone) and the card has room (< 8 GB used: a bev eval may run next to us)
while pgrep -f "run.py --id t11.vel_hurdle_acc2" > /dev/null; do sleep 30; done
ok=0; while [ $ok -lt 2 ]; do if [ "$(gpu_used)" -lt 8000 ]; then ok=$((ok + 1)); else ok=0; fi; sleep 30; done
echo "GPU free enough $(date +%T): $(gpu_used) MiB used"
python3 ../autoresearch/tools/run.py --id E266.b0 --gpu --require-card --split val \
  -- bash tools/env/_py14.sh evaluate.py recall_ring --compare recall > temp/loop22/E266b0.launcher.log 2>&1
echo "E266.b0: $(tail -n 1 temp/loop22/E266b0.launcher.log) $(date +%T)"
python3 ../autoresearch/tools/run.py --id E266.g01 --require-card --split val \
  -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 bash tools/env/_py14.sh experiments/E265_far-coverage-loss-budget/ring_split.py recall_ring recall > temp/loop22/E266g01.launcher.log 2>&1
echo "E266.g01: $(tail -n 1 temp/loop22/E266g01.launcher.log) $(date +%T)"
