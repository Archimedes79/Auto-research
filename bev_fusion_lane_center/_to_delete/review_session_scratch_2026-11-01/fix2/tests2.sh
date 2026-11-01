export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=/home/ml/miniconda3/envs/gcar/bin/python
cd /mnt/c/Auto-research/autoresearch
$PY -m evalsuite selftest > /tmp/f2_evalsuite.txt 2>&1; echo "evalsuite exit $?"; tail -1 /tmp/f2_evalsuite.txt
$PY -m trainsuite selftest > /tmp/f2_trainsuite.txt 2>&1; echo "trainsuite exit $?"; tail -1 /tmp/f2_trainsuite.txt
cd /mnt/c/Auto-research/bev_fusion_lane_center
bash tools/env/_py14.sh test_evaluate_guard.py 2>&1 | tail -1
cd /mnt/c/Auto-research/bev_fusion
$PY evaluation/bev_selftest.py > /tmp/f2_bev.txt 2>&1; echo "bev_selftest exit $?"; tail -1 /tmp/f2_bev.txt
$PY tests/test_gt_cache_targets.py 2>&1 | tail -1
$PY -c "import ast,sys; [ast.parse(open(p).read()) for p in ('training/train.py','training/nuscenes_dataset.py')]; print('syntax ok')"
$PY training/train.py > /tmp/f2_train.txt 2>&1; echo "train selftest exit $?"; tail -2 /tmp/f2_train.txt
