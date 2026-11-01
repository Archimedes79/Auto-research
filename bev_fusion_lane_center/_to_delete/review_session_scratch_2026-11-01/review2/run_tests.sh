cd /mnt/c/Auto-research/bev_fusion_lane_center
export BFLC_ROOT=$PWD
source tools/env/bflc_env.sh
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
P=/home/ml/miniconda3/envs/gcar/bin/python
cd evaluation/tests; $P test_lane_task.py 2>&1 | tail -3; $P test_evaluate_guard.py 2>&1 | tail -2
cd ../sparse_lanes; $P selftest.py 2>&1 | tail -3; $P topo_metrics.py 2>&1 | tail -2
cd ../lane_grid; timeout 300 $P selftest.py 2>&1 | tail -3
cd /mnt/c/Auto-research/autoresearch; timeout 400 $P -m evalsuite.cli selftest 2>&1 | tail -4; timeout 300 $P trainsuite/selftest.py 2>&1 | tail -3
