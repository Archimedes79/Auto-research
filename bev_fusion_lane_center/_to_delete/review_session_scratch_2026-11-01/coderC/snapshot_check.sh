#!/usr/bin/env bash
# 1 a snapshot of the CURRENT training/ (as run_cfg.sh makes it, runs/<name>/code) finds autogt three levels up without gt_bev.py
# 2 the pinned commits of the running lanes carry their own gt_bev.py
# run from the bev_fusion folder
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=/home/ml/miniconda3/envs/gcar/bin/python
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/coderC/snap
rm -rf $S && mkdir -p $S/runs/x/code
ln -s "$(pwd)/autogt" $S/autogt
cp training/*.py $S/runs/x/code/
rm -f $S/runs/x/code/gt_bev.py
cd $S/runs/x/code && $PY -c 'import nuscenes_dataset, losses, feature_cache, train, model; import autogt, os; print("snapshot without gt_bev.py imports ok; autogt from", os.path.dirname(autogt.__file__))'
cd - > /dev/null
for rev in 96a02be 4719d1d; do echo "$rev: $(git ls-tree --name-only $rev training/ | grep -c gt_bev.py) gt_bev.py in training/"; done
rm -rf $S
