cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/c/Auto-research
PY=~/miniconda3/envs/gcar/bin/python
nice -n 15 $PY evaluation/bev_junction_check.py 2>&1 | tail -5
nice -n 15 $PY evaluation/bev_junction_check.py --reproduce 2>&1 | tail -7
