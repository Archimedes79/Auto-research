cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
/home/ml/miniconda3/envs/gcar/bin/python training/train.py > /tmp/t11probe/train_selftest.log 2>&1
echo EXIT $? >> /tmp/t11probe/train_selftest.log
