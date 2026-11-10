#!/bin/bash
# word-level merge of the conflicted files of the cherry-pick in the integration worktree
W=/c/Auto-research/.claude/worktrees/integration
S=/c/Auto-research/bev_fusion/temp/cleanup/tm
mkdir -p $S
cd $W || exit 1
for f in bev_fusion/README.md bev_fusion/data/README.md bev_fusion/records/README.md bev_fusion/tests/README.md bev_fusion/training/README.md bev_fusion/training/architecture.json bev_fusion/training/nuscenes_dataset.py bev_fusion/tests/test_grad_scaler.py; do
  n=$(echo $f | tr '/' '_')
  git show :1:$f > $S/$n.b 2>/dev/null || { echo "no base for $f"; : > $S/$n.b; }
  git show :2:$f > $S/$n.o
  git show :3:$f > $S/$n.t
done
