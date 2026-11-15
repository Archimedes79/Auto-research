#!/usr/bin/env bash
cd /mnt/c/Auto-research || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
F="$(git ls-files 'bev_fusion/studies/*.py' 'bev_fusion/tests/*.py' 'bev_fusion/tests/support/*.py') bev_fusion/training/precompute.py bev_fusion/training/nuscenes_dataset.py bev_fusion/training/augment.py bev_fusion/training/feature_cache.py bev_fusion/training/cache_bevfusion_features.py bev_fusion/training/scratch.py bev_fusion/training/freeze.py bev_fusion/training/probe.py bev_fusion/training/debug.py bev_fusion/training/ab_timing.py bev_fusion/training/splits.py bev_fusion/training/train_motion_block.py"
$PY bev_fusion/temp/cleanup/audit_b/dead_funcs.py $F > /tmp/audit_b_dead.txt 2>&1
cat /tmp/audit_b_dead.txt
