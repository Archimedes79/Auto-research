#!/bin/bash
# scratch: check and commit the E276 head code by path
cd /mnt/c/Auto-research || exit 1
P=bev_fusion_lane_center
FILES="$P/HANDOVER.md $P/evaluation/README.md $P/evaluation/qgraph_eval.py $P/evaluation/qgraph_run_eval.py $P/evaluation/tests/test_tier_metrics.py $P/training/data/lane_gt_data.py $P/training/grid/README.md $P/training/grid/qgraph.py $P/training/grid/qtargets.py $P/training/grid/test_qtargets.py $P/training/grid/test_tier_loss.py $P/training/grid/test_tiers_sparse.py $P/training/train/qgates.py $P/training/train/qgraph_train.py $P/experiments/E276_gt-tiers-sparse-nodes-tier-loss/experiment.md"
python3 autoresearch/tools/check_git.py --paths $FILES || exit 2
python3 autoresearch/tools/commit_paths.py -F $P/temp/e276/commit_msg2.txt $FILES
echo COMMIT=$?
git log --oneline -2
