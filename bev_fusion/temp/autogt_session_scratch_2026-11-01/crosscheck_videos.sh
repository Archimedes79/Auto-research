#!/bin/bash
# scratch: the four crosscheck videos from trainval sequences of the four map areas (the code of the clean worktree temp/gt_head = HEAD + the data / video changes)
cd "$(dirname "$0")/../../../temp/gt_head" || exit 1
OUT=/mnt/c/Auto-research/data/autogt/examples
mkdir -p $OUT
for s in scene-0441 scene-1053 scene-0221 scene-0995; do
  echo "== $s"
  bash autogt/run_nice.sh -m autogt video $s $OUT/crosscheck_$s.mp4 20 --version v1.0-trainval 2>&1 | tail -3
done
