#!/usr/bin/env bash
# import checks of the training modules in script mode and package mode (run from the bev_fusion folder)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=/home/ml/miniconda3/envs/gcar/bin/python
$PY tests/test_gt_cache_targets.py 2>&1 | tail -n 5
(cd training && $PY -c 'import feature_cache, train, model, losses, nuscenes_dataset, augment, gt_cache_build; print("script-mode imports ok")')
$PY -c 'import training.train, training.feature_cache; print("package-mode imports ok")'
