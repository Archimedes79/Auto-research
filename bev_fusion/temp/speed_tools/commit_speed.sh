#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F temp/msg_speed_host_work.txt \
  bev_fusion/training/train.py bev_fusion/training/probe.py bev_fusion/training/losses.py bev_fusion/training/object_head.py \
  bev_fusion/tests/test_speed_flags.py bev_fusion/tests/test_host_overhead.py bev_fusion/tests/run_gates.sh \
  --blob bev_fusion/training/README.md=bev_fusion/temp/blob_training_README.md \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/blob_tests_README.md "$@" 2>&1 | tail -20
