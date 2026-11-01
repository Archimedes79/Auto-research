# lane GPU window part 2 (loop 22): E266.b1 (also E267's seed twin), then E267.sr (CPU) in parallel with E266.b2 / b3 (GPU), then E266.v (CPU)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
echo 1000 > /proc/self/oom_score_adj
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
gpu() {  # gpu <id> <name> <compare>
  python3 ../autoresearch/tools/run.py --id $1 --gpu --require-card --split val \
    -- bash tools/env/_py14.sh evaluate.py $2 --compare $3 > temp/loop22/$1.launcher.log 2>&1
  echo "$1: $(tail -n 1 temp/loop22/$1.launcher.log) $(date +%T)"; }
gpu E266.b1 recall_s1_ring recall_s1
( python3 ../autoresearch/tools/run.py --id E267.sr --require-card --split val \
    -- $T bash -c 'bash tools/env/_py14.sh evaluate.py recall_ring --asm spur_len=0 --tag sp0 --compare recall && bash tools/env/_py14.sh evaluate.py recall_s1_ring --asm spur_len=0 --tag sp0 --compare recall_s1' > temp/loop22/E267sr.launcher.log 2>&1
  echo "E267.sr: $(tail -n 1 temp/loop22/E267sr.launcher.log) $(date +%T)" ) &
sleep 60   # no simultaneous run.py starts
gpu E266.b2 fast_recall_ring fast_recall
gpu E266.b3 e24_j1rc_s1_ring e24_j1rc_s1
wait
python3 ../autoresearch/tools/run.py --id E266.v --require-card --split val \
  -- $T bash tools/env/_py14.sh experiments/E265_far-coverage-loss-budget/cleanup_split.py recall_ring recall_s1_ring fast_recall_ring e24_j1rc_s1_ring > temp/loop22/E266v.launcher.log 2>&1
echo "E266.v: $(tail -n 1 temp/loop22/E266v.launcher.log) $(date +%T)"
