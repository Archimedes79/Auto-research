cd /mnt/c/Auto-research/bev_fusion_lane_center; export BFLC_ROOT=$PWD; source tools/env/bflc_env.sh
cd evaluation/lane_grid; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 timeout 300 /home/ml/miniconda3/envs/gcar/bin/python selftest.py 2>&1 | head -8
