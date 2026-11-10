#!/bin/bash
# scratch: park my duplicate runner draft (moved, nothing deleted) and restore the runner README
cd /mnt/c/Auto-research/bev_fusion || exit 1
D=temp/cleanup/runner_draft
mkdir -p $D
mv runner/resultset.py runner/bev_results.py tests/test_resultset.py tests/test_bev_results.py $D/ || exit 1
mv runs/t20_base_s0/results/75345872a4 $D/results_75345872a4 || exit 1
rmdir runs/t20_base_s0/results
cp temp/cleanup/test_real.log $D/test_real.log
git -C /mnt/c/Auto-research restore -- bev_fusion/runner/README.md
git -C /mnt/c/Auto-research status --short -- bev_fusion/runner bev_fusion/tests | cat
ls $D
ls runs/t20_base_s0
