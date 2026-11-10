#!/bin/bash
PY=~/miniconda3/envs/gcar/bin/python
echo "--- from bev_fusion: from runner import Runner"
(cd /mnt/c/Auto-research/bev_fusion && $PY -c "from runner import Runner; print('ok', Runner)" 2>&1 | tail -2)
echo "--- from bev_fusion: from runner.runner import Runner"
(cd /mnt/c/Auto-research/bev_fusion && $PY -c "from runner.runner import Runner; print('ok', Runner)" 2>&1 | tail -2)
echo "--- from autoresearch: from runner.runner import Runner"
(cd /mnt/c/Auto-research/autoresearch && $PY -c "from runner.runner import Runner; print('ok', Runner)" 2>&1 | tail -2)
echo "--- from anywhere with PYTHONPATH=bev_fusion"
(cd /tmp && PYTHONPATH=/mnt/c/Auto-research/bev_fusion $PY -c "from runner.runner import Runner; print('ok', Runner)" 2>&1 | tail -2)
echo "--- system python3 (no numpy/conda)"
(cd /mnt/c/Auto-research/bev_fusion && python3 -c "from runner.runner import Runner; print('ok')" 2>&1 | tail -2)
