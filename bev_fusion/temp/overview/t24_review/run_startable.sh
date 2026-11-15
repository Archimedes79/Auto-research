cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
for c in t21_ctl t24_lane_pull; do
  echo "=== startable $c"
  nice -n 12 $HOME/miniconda3/envs/gcar/bin/python training/startable.py training/configs/$c.json 2>&1 | cut -c1-300 | tail -n 12
done
echo done
