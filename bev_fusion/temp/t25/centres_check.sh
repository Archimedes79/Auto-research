#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
echo "== lift_pd selftest"; nice -n 15 $PY training/lift_pd.py 2>&1 | grep -v '^WARNING' | tail -n 3 | cut -c1-200
echo "== same cell centres as GtGenerator.grid.GridSpec"
nice -n 15 $PY - <<'PYEOF' 2>&1 | tail -n 3
import sys
sys.path.insert(0, "training")
import numpy as np
from lift_pd import Grid
from GtGenerator.grid import GridSpec
g = Grid()
a = g.centres(); b = GridSpec(g.x_min, g.x_max, g.y_min, g.y_max, g.res).centres()
print("equal:", all(np.array_equal(x, y) for x, y in zip(a, b)))
PYEOF
echo "== dataset selftest with GT (was broken)"
nice -n 15 $PY training/nuscenes_dataset.py 2>&1 | grep -v '^WARNING' | tail -n 4 | cut -c1-220
