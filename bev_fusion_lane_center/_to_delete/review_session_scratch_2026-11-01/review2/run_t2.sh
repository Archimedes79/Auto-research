cd /mnt/c/Auto-research/autoresearch
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
timeout 300 /home/ml/miniconda3/envs/gcar/bin/python -m trainsuite.selftest 2>&1 | tail -4
