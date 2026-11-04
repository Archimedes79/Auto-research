#!/bin/bash
# a pinned copy of the run snapshot's autogt (v5, signature 15efbde2d1) for the resume of t10_dual_long_s0, so a later GT version in the live autogt cannot reach the run:
# runs/t10_dual_long_s0/autogt is found first by the walk-up of nuscenes_dataset / losses / gt_bev (code/ -> runs/t10_dual_long_s0/ -> ...)
cd "$(dirname "$0")/../.." || exit 1
snap=d14597e2e8
dest=runs/t10_dual_long_s0
if [ -d $dest/autogt ]; then echo "already there: $dest/autogt"; else
  git -C .. archive "$snap" bev_fusion/autogt | tar -x -C $dest --strip-components=1 || exit 1
  echo "extracted bev_fusion/autogt of $snap into $dest/autogt"
fi
ls $dest/autogt | tr '\n' ' ' | fold -w 200; echo
echo "---- which autogt does the trainer's import find? (cwd bev_fusion, no PYTHONPATH, the snapshot copy of training/ first on the path, as in the run)"
cd runs/t10_dual_long_s0/code || exit 1
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 ~/miniconda3/envs/gcar/bin/python -c "
import sys, os
sys.argv = ['x']
os.chdir('../../..')
sys.path.insert(0, os.path.abspath('runs/t10_dual_long_s0/code'))
import nuscenes_dataset, losses
import autogt
print('autogt from', os.path.dirname(os.path.abspath(autogt.__file__)))
from autogt import nuscenes_gt
print('nuscenes_gt from', os.path.abspath(nuscenes_gt.__file__))
print('signature files:', [os.path.basename(f) for f in nuscenes_gt.SIGNATURE_FILES])
" 2>&1 | tail -5 | cut -c1-200
