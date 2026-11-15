#!/bin/bash
# scratch (Image Encoder, clean-base round): commit of the B2 removal, own paths, temporary index; the shared files go in as blobs (HEAD + my hunk); "dry" as $1 = dry run
cd /mnt/c/Auto-research || exit 1
H=bev_fusion/temp/cleanup/headcopy
CLEAN="bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py bev_fusion/tests/test_student_chain.py bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/training/configs/t22_B2_s0.json"
SHARED="bev_fusion/training/train.py bev_fusion/training/startable.py bev_fusion/training/README.md bev_fusion/training/configs/README.md bev_fusion/tests/README.md"
# the clean files must still be exactly what I tested: nobody else touched them since the stat I recorded
for f in bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py; do
  echo "$f: $(git diff --numstat -- $f | cut -f1,2)"
done
python3 autoresearch/tools/check_git.py --paths $CLEAN $SHARED 2>&1 | tail -8
echo "--- check_git exit: ${PIPESTATUS[0]}"
BLOBS=""
for f in $SHARED; do BLOBS="$BLOBS --blob $f=$H/$f"; done
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_msg_clean.txt $CLEAN $BLOBS $EXTRA 2>&1 | tail -16
git log --oneline -2 | cut -c1-160
