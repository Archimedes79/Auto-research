cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2"
# chain 1: E264.u (u01, u4 ceiling, urc no-diversity control) then E264.v (per-piece DP of the unions)
( python3 ../autoresearch/tools/run.py --id E264.u --require-card --split val \
    -- $T bash -c 'bash tools/env/_py14.sh evaluate.py ens_j1rc_01 --compare e24_j1rc_s1 && bash tools/env/_py14.sh evaluate.py ens_j1rc_0123 --compare e24_j1rc_s1 && bash tools/env/_py14.sh evaluate.py ens_rc1 --compare recall' > temp/loop22/E264u.launcher.log 2>&1
  echo "E264.u: $(tail -n 1 temp/loop22/E264u.launcher.log)"
  python3 ../autoresearch/tools/run.py --id E264.v --require-card --split val \
    -- $T bash tools/env/_py14.sh experiments/E265_far-coverage-loss-budget/cleanup_split.py ens_j1rc_23 ens_j1rc_01 > temp/loop22/E264v.launcher.log 2>&1
  echo "E264.v: $(tail -n 1 temp/loop22/E264v.launcher.log)" ) &
sleep 60   # no simultaneous run.py starts
# chain 2: E264.mfl (val250 tau ladder of the parents + union val250 cards)
python3 ../autoresearch/tools/run.py --id E264.mfl --require-card --split val \
  -- $T bash -c 'bash tools/env/_py14.sh evaluate.py ens_j1rc_23 --subset val250 --tag u250 && for t in 0.40 0.45 0.50 0.55 0.60 0.65; do bash tools/env/_py14.sh evaluate.py e24_j1rc_s3 --subset val250 --tau $t --tag t$t || exit 1; done && bash tools/env/_py14.sh evaluate.py ens_j1rc_01 --subset val250 --tag u250 && for t in 0.40 0.45 0.50 0.55 0.60 0.65; do bash tools/env/_py14.sh evaluate.py e24_j1rc_s1 --subset val250 --tau $t --tag t$t || exit 1; done' > temp/loop22/E264mfl.launcher.log 2>&1
echo "E264.mfl: $(tail -n 1 temp/loop22/E264mfl.launcher.log)"
wait
