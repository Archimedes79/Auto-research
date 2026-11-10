#!/bin/bash
# the commands of records/jobs/chain/README.md that can be tried without starting anything
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
echo "=== list"; python3 -m jobrunner JobRunner.list --project ../bev_fusion 2>&1 | tail -12
echo "=== status chain2.score"; python3 -m jobrunner JobRunner.status chain2.score --project ../bev_fusion 2>&1 | head -25
echo "=== sets"; ls /mnt/c/Auto-research/GtGenerator/sets/; (cd /mnt/c/Auto-research && python3 -m GtGenerator sets 2>&1 | head -12)
echo "=== show-numbers help"; $HOME/miniconda3/envs/gcar/bin/python -m evalsuite methods 2>&1 | head -30
