#!/bin/bash
# scratch: the commit of the stage-B chain step (own paths only), $1 = --dry-run or empty
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/chain/commit_msg.txt $1 \
  bev_fusion/training/student_encoder.py bev_fusion/training/distill_student.py bev_fusion/training/model.py bev_fusion/training/feature_cache.py \
  bev_fusion/training/train.py bev_fusion/training/interface.py bev_fusion/training/startable.py bev_fusion/training/README.md \
  bev_fusion/training/architecture.json bev_fusion/training/architecture.md \
  bev_fusion/tests/test_student_chain.py bev_fusion/tests/test_slots.py bev_fusion/tests/test_trainkit_equivalence.py bev_fusion/tests/test_velocity_hurdle.py \
  bev_fusion/tests/README.md bev_fusion/tests/run_gates.sh
