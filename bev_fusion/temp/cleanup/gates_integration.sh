#!/bin/bash
# fast gates of bev_fusion in the integration worktree (clean tree: HEAD + the two coder commits)
W=/mnt/c/Auto-research/.claude/worktrees/integration
R=/mnt/c/Auto-research/bev_fusion
cd $W/bev_fusion || exit 1
export GIT_DIR=/mnt/c/Auto-research/.git/worktrees/integration GIT_WORK_TREE=$W
# read-only links to what is not in git
mkdir -p data runs models/pretrained
[ -e data/cache ] || ln -s $R/data/cache data/cache
for r in t20_base_s0 t16_full_s0 t10_dual_s0; do [ -e runs/$r ] || ln -s $R/runs/$r runs/$r; done
[ -e models/pretrained/dualbev-r50-cbgs.pth ] || ln -s $R/models/pretrained/dualbev-r50-cbgs.pth models/pretrained/dualbev-r50-cbgs.pth
[ -d models/pretrained/imagenet ] || { [ -d $R/models/pretrained/imagenet ] && ln -s $R/models/pretrained/imagenet models/pretrained/imagenet; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
date +%T
nice -n 10 bash tests/run_gates.sh 2>&1 | tee $R/temp/cleanup/gates_integration.log
date +%T
