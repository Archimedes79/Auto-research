cd /mnt/c/Auto-research/bev_fusion_lane_center
echo 1000 > /proc/self/oom_score_adj
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
# prototype candidate S card first (assembly-only, cached raw outputs)
bash tools/env/_py14.sh evaluate.py recall_sp0_dp --compare recall_dp > temp/loop22/recall_sp0_dp.eval.log 2>&1
echo "recall_sp0_dp: $(tail -n 2 temp/loop22/recall_sp0_dp.eval.log)"
# E266.gold: layer 0 / 1 after the loop-22 code
python3 ../autoresearch/tools/run.py --id E266.gold --require-card --split val \
  -- bash -c 'bash tools/env/_py14.sh evaluate.py --golden check --cached-only && bash tools/maintenance/run_all_tests.sh' > temp/loop22/E266gold.launcher.log 2>&1
echo "E266.gold: $(tail -n 1 temp/loop22/E266gold.launcher.log)"
# E264.u23: held-out union card (CPU, cached raw of both members)
python3 ../autoresearch/tools/run.py --id E264.u23 --require-card --split val \
  -- bash tools/env/_py14.sh evaluate.py ens_j1rc_23 --compare e24_j1rc_s3 > temp/loop22/E264u23.launcher.log 2>&1
echo "E264.u23: $(tail -n 1 temp/loop22/E264u23.launcher.log)"
