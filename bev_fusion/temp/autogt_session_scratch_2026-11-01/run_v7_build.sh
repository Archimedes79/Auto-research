#!/bin/bash
# scratch: the one trainval build under GT version v7 (signature b84e8ac9b2): the code of the tag autogt/v7 in the worktree temp/gt_v7 (pinned and clean,
# so the manifest can say release), the output into the shared data/autogt. Resumable: run it again after an interruption.
cd /mnt/c/Auto-research/temp/gt_v7 || exit 1
exec bash autogt/run_nice.sh -m autogt generate '~/data/nuscenes_full' --set full --out /mnt/c/Auto-research/data/autogt >> /mnt/c/Auto-research/bev_fusion/temp/autogt_session_scratch_2026-11-01/v7_build.log 2>&1
