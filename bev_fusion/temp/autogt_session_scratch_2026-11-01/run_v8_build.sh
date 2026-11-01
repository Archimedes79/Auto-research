#!/bin/bash
# scratch: the one trainval build under GT version v8 (signature c643a7539d): the code of the tag autogt/v8 in the worktree temp/gt_v8 (pinned and
# clean, so the manifest can say release), two worker processes, the output into the shared data/autogt. Resumable: run it again after an interruption.
cd /mnt/c/Auto-research/temp/gt_v8 || exit 1
exec bash autogt/run_nice.sh -m autogt generate '~/data/nuscenes_full' --set full --workers 2 --out /mnt/c/Auto-research/data/autogt >> /mnt/c/Auto-research/bev_fusion/temp/autogt_session_scratch_2026-11-01/v8_build.log 2>&1
