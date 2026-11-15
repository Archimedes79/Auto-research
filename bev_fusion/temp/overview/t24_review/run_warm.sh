cd /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
nice -n 12 $HOME/miniconda3/envs/gcar/bin/python probe_warm.py 2>&1 | tail -n 20
echo done
