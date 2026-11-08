cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in tests/test_lane_pull.py tests/test_frozen.py tests/test_lane_branch_dual.py training/lane_branch.py training/lift_pd.py; do
  echo "=== $t"
  nice -n 12 $PY $t 2>&1 | tail -n 60
  echo "exit ${PIPESTATUS[0]}"
done
