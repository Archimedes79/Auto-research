cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/gcar/bin/python
for c in t21_ctl t24_lane_pull; do
  $PY training/splits.py resolve < training/configs/$c.json > temp/overview/t24_review/res_$c.json
  echo "=== startable (resolved) $c"
  nice -n 12 $PY training/startable.py temp/overview/t24_review/res_$c.json 2>&1 | cut -c1-260 | grep -v "^start "
done
echo done
