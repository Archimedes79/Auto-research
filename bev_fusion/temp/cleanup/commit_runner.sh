#!/bin/bash
# scratch: commit the runner/ unit through a temporary index (own paths only; tests/README.md as a blob built from HEAD)
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/cleanup/blobs_runner.py || exit 1
P="bev_fusion/runner/runner.py bev_fusion/runner/__main__.py bev_fusion/runner/README.md bev_fusion/README.md \
bev_fusion/tests/test_runner_class.py bev_fusion/tests/test_runner_real.py bev_fusion/tests/test_suites_conform.py bev_fusion/tests/run_gates.sh"
git status --short -- $P bev_fusion/tests/README.md | cat
python3 autoresearch/tools/check_git.py --paths $P bev_fusion/tests/README.md || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_runner_msg.txt $P \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/cleanup/blobs/bev_fusion_tests_README.md || exit 1
git log --oneline -1 | cut -c1-140
git status --short -- $P bev_fusion/tests/README.md | cat
