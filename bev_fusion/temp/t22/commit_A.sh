#!/bin/bash
# scratch: commit A of the t22 refinement C: code, tests, card, configs, README rows (HEAD + my rows only, built by readme_edit.py) through a temporary index; "dry" as $1 = print only
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/training/feature_cache.py bev_fusion/training/interface.py bev_fusion/training/model.py bev_fusion/training/startable.py bev_fusion/training/train.py bev_fusion/training/lane_branch.py
   bev_fusion/training/configs/t22_B2_s0.json bev_fusion/training/configs/README.md
   bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py bev_fusion/evaluation/stream_drift.py
   bev_fusion/tests/test_student_chain.py bev_fusion/tests/test_student_encoder.py bev_fusion/tests/test_stream_tools.py
   bev_fusion/experiments/t22_student_encoder/experiment.md"
python3 bev_fusion/temp/t22/rebuild_readme_blobs.py || exit 1          # the README copies from the current HEAD + my rows only (another chat may have committed rows meanwhile)
python3 autoresearch/tools/check_git.py --paths $P bev_fusion/tests/README.md bev_fusion/training/README.md 2>&1 | tail -12
echo "--- check_git exit: ${PIPESTATUS[0]}"
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_A.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/t22/blob_tests_README.md --blob bev_fusion/training/README.md=bev_fusion/temp/t22/blob_training_README.md $EXTRA 2>&1 | tail -30
